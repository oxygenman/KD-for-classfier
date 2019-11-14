
## Quick Start


##  Dependency

* Python 3.6
* Pytorch 1.0
* tqdm (pip install tqdm)

### Note
* Hyper-parameters that used for experiments in the paper are specified at scripts in ```exmples/```.
* Heavy teacher network (ResNet50 w/ 512 dimension) requires more than 12GB of GPU memory if batch size is 128.  
  Thus, you might have to reduce the batch size. (The experiments in the paper were conducted on P40 with 24GB of gpu memory. 
)
