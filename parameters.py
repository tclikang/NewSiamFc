class param:
    def __init__(self):
        self.prior_frames_num = 6
        self.regression_loss_weight = 2
        self.class_weight = 1
        self.model_save_path = '/home/fanfu/PycharmProjects/SimpleSiamFC/pretrained/siamfc_new/'
        self.kernel_lr = 0.975  # 0.975 two loss better
        self.update_template_threshold = 0.75  # 当前响应图占第一帧响应图最大值的75%,才更新