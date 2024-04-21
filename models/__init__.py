import importlib
from models.base_model import BaseModel
import os


def find_model_class_by_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print(
            "In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase."
            % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    model_class = find_model_class_by_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    model = find_model_class_by_name(opt.model)
    instance = model()
    instance.initialize(opt)
    print("model [{}] was created".format(instance.name()))
    return instance

def load_model(opt):
    print(opt.checkpoints_dir + opt.name + "/*_net_ray_marching.pth")
    # FeatureNet = FN(intermediate=True).to(device)
    # new_state_dict_fn = load_pretrained(os.path.join(opt.resume_dir, 'best_net_mvs.pth'))
    # FeatureNet.load_state_dict(new_state_dict_fn)

    resume_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if opt.resume_iter == "best":
        opt.resume_iter = "latest"
    resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(resume_dir)
    if resume_iter is None:
        print("No previous checkpoints at iter {} !!", resume_iter)
        exit()
    else:
        opt.resume_iter = resume_iter
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('test at {} iters'.format(opt.resume_iter))
        print(f"Iter: {resume_iter}")
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    opt.mode = 2
    opt.load_points=1
    opt.resume_dir=resume_dir
    opt.resume_iter = resume_iter

    model = create_model(opt)
    model.setup(opt)

    return model