import configs.options

class Config(object):
    def __init__(self, args):
        self.root_src = args.root_src     # 根目录
        self.root_full = args.root_full       # 根目录
        self.modal = args.modal       # 模型名字
        self.lr = args.lr      # 学习率，eval字符串转数学运算
        self.num_iters = args.num_iters         # 迭代次数
        self.len_feature = 1024       # 特征长度   I3D提取的特征是时间长度*特征长度
        self.batch_size = args.batch_size      # 
        self.weight_path = args.weight_path    # 模型预训练权重地址
        self.output_path = args.output_path
        self.num_workers = args.num_workers      # 多少cpu
        self.model_file = args.model_file
        self.seed = args.seed
        self.num_segments = args.num_segments

            
if __name__ == "__main__":
    args=options.parse_args()
    conf=Config(args)
    print(conf.lr)  

