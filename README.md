# Clothes-AWS

### Diff
- removed torch==19.1+cu111 dependency, currently intalling without specifying version
- issue with memory size on aws ebs, changing instace to t2.small ans t2.medium
- t2.small is also insufficient, switching to t2.medium 
