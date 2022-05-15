# ReID_maket1501

A tiny repo Re-implementing a solution for reID problem. This project refers the official code [link](https://github.com/michuanhaohao/reid-strong-baseline) and can reproduce the results as good as it on Market1501 when the input size is set to 256x128

Develop based on the [pytorch template](https://github.com/lulujianjie/pytorch-project-template)

## Directory layout

    .
    ├── config                  # hyperparameters settings
    │   └── ...                 
    ├── datasets                # data loader
    │   └── ...           
    ├── log                     # log and model weights
    |   └── ...              
    ├── loss                    # loss function code
    │   └── ...   
    ├── model                   # model (nnet, backbone)
    │   └── ...  
    ├── processor               # training and testing procedures
    │   └── ...    
    ├── solver                  # optimization code
    │   └── ...   
    ├── tools                   # tools
    │   └── ...
    ├── utils                   # metrics code
    │   └── ...
    ├── train.py                # train code 
    ├── test.py                 # test code 
    ├── get_vis_result.py       # get visualized results 
    ├── docs                    # docs for readme              
    └── README.md