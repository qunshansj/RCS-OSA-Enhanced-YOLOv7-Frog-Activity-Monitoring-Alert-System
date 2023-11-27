python


class ObjectDetection:
    def __init__(self, opt):
        self.opt = opt
        self.names = opt.names
        self.source = opt.source
        self.weights = opt.weights
        self.view_img = opt.view_img
        self.save_txt = opt.save_txt
        self.imgsz = opt.img_size
        self.trace = not opt.no_trace
        self.save_img = not opt.nosave and not source.endswith('.txt')
        self.webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        self.save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (self.save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # initialize deepsort
        cfg_deep = get_config()
        cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        self.point_list = []
        # Initialize
        set_logging()
        self.device = select_device(opt.device)
        self.half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        self.stride = int(model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            self.model = TracedModel(model, device, opt.img_size)

        if half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        self.vid_path, self.vid_writer = None, None
        if webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            self.dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        self.names = load_classes(names)
        #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1

        self.t0 = time.time()

    def detect(self):
        for path, img, im0s, vid_cap in self.dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # ...

        return img
