# ReID_maket1501

A tiny repo Re-implementing a solution for reID problem. 
This project reproduce the SSKD code [link](https://github.com/xiaomingzhid/sskd) with some change base on my own datasets and resource.


## Directory layout

    .
    ├── configs                  # hyperparameters settings
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