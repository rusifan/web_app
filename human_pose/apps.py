from django.apps import AppConfig

class HumanPoseConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'human_pose'
    # MODEL_PATH = Path("model")
    # state_dict = os.path.join(MODEL_PATH, "model_5.pth")
    # new_state_dict = OrderedDict()
    
    # for k, v in state_dict.items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)