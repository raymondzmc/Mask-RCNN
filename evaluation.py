import os, torch, skimage, pdb
from config import Config
import model as modellib
import utils
from skimage.transform import resize

class FashionConfig(Config):
    NAME = 'fashion'

    # I don't have enough memory on my GPU
    GPU_COUNT = 0
    # IMAGES_PER_GPU = 1

    # Since we are using COCO pretrained weights
    NUM_CLASSES = 1 + 80

if __name__ == "__main__":
    CUR_DIR = os.getcwd()

    # Initialize model
    config = FashionConfig()
    config.display()
    MODEL_DIR = os.path.join(CUR_DIR, "logs")
    model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)

    # if torch.cuda.is_available():
    #     model = model.cuda()

    # Download pretrained weights
    coco_pretrained = os.path.join(CUR_DIR, 'mask_rcnn_coco.pth')
    if not os.path.isfile(coco_pretrained):
        print("Downloading Pretrained Model...")
        share_id = '1RhdD8PkR_AQ1-uP3JS-nbcXaTOILhXua'
        utils.download_file_from_google_drive(share_id, coco_pretrained)

    model.load_state_dict(torch.load(coco_pretrained))

    data_path = os.path.join(CUR_DIR, 'images')

    results_path = os.path.join(CUR_DIR, 'results')

    img_list = [f for f in os.listdir(data_path) if '.jpg' in f]

    for img_name in img_list:
        save_name = os.path.join(results_path, img_name.split('.')[0] + '.pth')
        if os.path.isfile(save_name):
            continue
        img_path = os.path.join(data_path, img_name)

        try:
            image = skimage.io.imread(img_path)
            image = resize(image, (224, 224), anti_aliasing=True)
        except:
            pdb.set_trace()

        results = model.detect([image])
        if len(results) != 1:
            pdb.set_trace()
        results = torch.from_numpy(results[0]['masks'])
        torch.save(results, save_name)
