import configs.options

class Config(object):
    def __init__(self, args):
        self.root_src = args.root_src     
        self.root_full = args.root_full      
        self.modal = args.modal      
        self.lr = args.lr     
        self.num_iters = args.num_iters        
        self.len_feature = 1024       
        self.batch_size = args.batch_size      
        self.weight_path = args.weight_path   
        self.output_path = args.output_path
        self.num_workers = args.num_workers     
        self.model_file = args.model_file
        self.seed = args.seed
        self.num_segments = args.num_segments

            
if __name__ == "__main__":
    args=options.parse_args()
    conf=Config(args)
    print(conf.lr)  

