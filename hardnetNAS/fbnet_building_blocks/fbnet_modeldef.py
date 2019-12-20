# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# please, end the file with '}' and nothing else. this file updated automatically

MODEL_ARCH = {
    "fbnet_cpu_sample2": {
            "block_op_type": [
            ["ir_k5_e6"], 
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["ir_k5_e6"], ["skip"], ["skip"], 
            ["ir_k5_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k5_e6"], ["skip"], ["skip"], ["skip"], 
            ["skip"], 
            ],
            "block_cfg": {
                "first": [16, 2],
                "stages": [
                    [[6, 16, 1, 1]],                                                            # stage 1
                    [[6, 24, 1, 2]],  [[6, 24, 1, 1]],      [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[6, 32, 1, 1]],      [[1, 32, 1, 1]],  [[1, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[6, 64, 1, 1]],      [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[6, 112, 1, 1]], [[1, 112, 1, 1]],     [[1, 112, 1, 1]], [[1, 112, 1, 1]], # stage 5
                    [[6, 184, 1, 2]], [[1, 184, 1, 1]],     [[1, 184, 1, 1]], [[1, 184, 1, 1]], # stage 6
                    [[1, 352, 1, 1]],                                                           # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },

    "wang2": {
            "block_op_type": [
            ["ir_k3_e1"], 
            ["ir_k5_e1"], 
            ["ir_k5_s2"], 
            ["ir_k3_s2"], 
            ["ir_k5_e1"], 
            ["skip"], 
               ],
               "block_cfg": {
                   "first": [16, 2],
                   "stages": [
                       [[1, 16, 1, 1]],                                                        # stage 1
                       [[1, 24, 1, 2]],   # stage 2
                       [[1, 32, 1, 2]],   # stage 3
                       [[1, 64, 1, 2]],   # stage 4
                       [[1, 112, 1, 1]],  # stage 5
                       [[1, 184, 1, 2]],  # stage 6
                   ],
                   "backbone": [num for num in range(23)],
               },
           },
    "wang3": {
            "block_op_type": [
            ["ir_k5_e1"], 
            ["skip"], 
            ["ir_k5_e1"], 
            ["skip"], 
            ["skip"], 
            ["skip"], 
               ],
               "block_cfg": {
                   "first": [16, 2],
                   "stages": [
                       [[1, 16, 1, 1]],                                                        # stage 1
                       [[1, 24, 1, 2]],   # stage 2
                       [[1, 32, 1, 2]],   # stage 3
                       [[1, 64, 1, 2]],   # stage 4
                       [[1, 112, 1, 1]],  # stage 5
                       [[1, 184, 1, 2]],  # stage 6
                   ],
                   "backbone": [num for num in range(23)],
               },
           },
    "wang4": {
            "block_op_type": [
            ["skip"], 
            ["skip"], 
            ["ir_k5_s2"], 
            ["ir_k3_s2"], 
            ["ir_k5_e1"], 
            ["ir_k5_e1"], 
               ],
               "block_cfg": {
                   "first": [16, 2],
                   "stages": [
                       [[1, 16, 1, 1]],                                                        # stage 1
                       [[1, 24, 1, 2]],   # stage 2
                       [[1, 32, 1, 2]],   # stage 3
                       [[1, 64, 1, 2]],   # stage 4
                       [[1, 112, 1, 1]],  # stage 5
                       [[1, 184, 1, 2]],  # stage 6
                   ],
                   "backbone": [num for num in range(23)],
               },
           },
   }   