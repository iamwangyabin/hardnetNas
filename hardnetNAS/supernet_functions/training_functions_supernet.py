import torch
from torch.autograd import Variable
import time
import numpy as np
from general_functions.utils import AverageMeter, save, accuracy, Meter
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from general_functions.EvalMetrics import ErrorRateAt95Recall
import general_functions.dataloader as dataloader

class TrainerSupernet:
    def __init__(self, criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer, lookup_table):
        self.logger = logger
        self.writer = writer

        self.criterion = criterion
        self.w_optimizer = w_optimizer
        self.theta_optimizer = theta_optimizer
        self.w_scheduler = w_scheduler

        self.temperature = CONFIG_SUPERNET['train_settings']['init_temperature']
        self.exp_anneal_rate = CONFIG_SUPERNET['train_settings']['exp_anneal_rate']  # apply it every epoch
        self.cnt_epochs = CONFIG_SUPERNET['train_settings']['cnt_epochs']
        self.train_thetas_from_the_epoch = CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch']
        self.print_freq = CONFIG_SUPERNET['train_settings']['print_freq']
        self.path_to_save_model = CONFIG_SUPERNET['train_settings']['path_to_save_model']

        self.lookup_table = lookup_table
        op_name = [op_name for op_name in self.lookup_table.lookup_table_operations]
        self.latency_table = []
        for i in self.lookup_table.lookup_table_latency:
            temp = []
            for j in op_name:
                temp.append(i[j])
            self.latency_table.append(temp)

    def train_loop(self, train_w_loader, train_thetas_loader, test_loader, model):

        best_top1 = 0.0
        best_lat = 9999.9

        # firstly, train weights only
        for epoch in range(self.train_thetas_from_the_epoch):
            # self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            self.logger.info("Prepare trainla %d" % (epoch))
            self._training_step(model, train_w_loader, self.w_optimizer)
            self.w_scheduler.step()

        for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):
            # self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            # self.writer.add_scalar('learning_rate/theta', self.theta_optimizer.param_groups[0]['lr'], epoch)
            self.logger.info("Epoch %d" % (epoch))
            self.logger.info("Train Weights for epoch %d" % (epoch))
            self._training_step(model, train_w_loader, self.w_optimizer)
            self.w_scheduler.step()
            self.logger.info("Train Thetas for epoch %d" % (epoch))
            self._training_step(model, train_thetas_loader, self.theta_optimizer)

            acc,lat = self._validate(model, test_loader)
            if acc>best_top1 or lat < best_lat:
                best_top1 = acc
                best_lat = lat
                self.logger.info("Best top1 acc by now. Save model")

                save(model, self.path_to_save_model+"acc%.3f_lat%.3f.pth"%(acc,lat))

            self.temperature = self.temperature * self.exp_anneal_rate
            train_w_loader = dataloader.create_loaders(load_random_triplets=False,
                                                       batchsize=CONFIG_SUPERNET['dataloading']['batch_size'],
                                                       n_triplets=50000)
            train_thetas_loader = dataloader.create_loaders(load_random_triplets=False,
                                                            batchsize=CONFIG_SUPERNET['dataloading']['batch_size'],
                                                            n_triplets=20000)

    def get_sample_latency(self, model):
        parameters = []
        sample_latency = 0
        for layer in model.module.stages_to_search:
            parameters.append(torch.argmax(layer.thetas).item())
        for i in range(len(parameters)):
            sample_latency += self.latency_table[i][parameters[i]]
        return sample_latency

    def _training_step(self, model, loader, optimizer):
        model = model.train()
        epochLoss = 0
        epochLat = 0
        epochCe = 0
        for step, (X, Y) in enumerate(loader):
            X, Y = X.cuda(non_blocking=True), Y.cuda(non_blocking=True)
            optimizer.zero_grad()
            latency_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda()
            outs_Y, latency_to_accumulate, soft1, hard1 = model(Y, self.temperature, latency_to_accumulate)
            outs_X, latency_to_accumulate, soft2, hard2 = model(X, self.temperature, latency_to_accumulate)
            soft_latency = (soft1 + soft2) / 2
            hard_latency = (hard1 + hard2) / 2

            # latency_to_accumulate/2 因为是用了两次
            loss, ce, lat = self.criterion(outs_X, outs_Y, latency_to_accumulate / 2, soft_latency)
            loss.backward()
            optimizer.step()

            epochLoss += loss
            epochLat += lat
            epochCe += ce
            if (step % self.print_freq == 0):
                self.logger.info('Step:%3d Loss:%.3f,Lat:%.3f,Ce:%.3f -- latency:%.4f, soft:%.4f, hard:%.4f' % (
                step, loss, lat, ce, latency_to_accumulate.item(), soft_latency.item(), hard_latency))

        self.logger.info("EPOCH Loss:%f,\tLat:%f,\tCe:%f" % (epochLoss / step, epochLat / step, epochCe / step))

    def _validate(self, model, loader):
        model.eval()
        accnum = 0
        latency = 0
        self.logger.info("Start Validate")
        with torch.no_grad():
            for step, (X, Y, label) in enumerate(loader):
                X, Y, label = X.cuda(), Y.cuda(), label.cuda()
                latency_to_accumulate = torch.Tensor([[0.0]]).cuda()
                outs_Y, latency_to_accumulate, _, hard1 = model(Y, self.temperature, latency_to_accumulate)
                outs_X, latency_to_accumulate, _, hard2 = model(X, self.temperature, latency_to_accumulate)
                latency += (hard1+hard2)/2
                for i in range(len(outs_X)):
                    if label[i] == 0:
                        continue
                    else:
                        dis = torch.sum((outs_Y - outs_X[i]) ** 2, 1)
                        top1value, top1 = torch.kthvalue(dis, 1)
                        top2value, top2 = torch.kthvalue(dis, 2)
                        if top1 == i and top2value * 0.8 > top1value:
                            accnum += 1
        accuracy = accnum / 50000
        latency = latency / step
        self.logger.info("accuracy:%f,\tsample_latency:%f" % (accuracy, latency))
        return accuracy, latency

    # def _intermediate_stats_logging(self, fpr95, epoch, val_or_train):
    #     self.top1.update(fpr95)
    #     self.logger.info(val_or_train + ": [{:3d}/{}] Loss {:.3f}, Prec@{:.4%}, ce_loss {:.3f}, lat_loss {:.3f}".format(
    #         epoch + 1, self.cnt_epochs, self.losses.get_avg(), fpr95, self.losses_ce.get_avg(),
    #         self.losses_lat.get_avg()))
