from PIL import Image
import numpy as np
import dehazing


def main():
    dehaze('./images/street.jpg')
    dehaze('./images/forest.jpg')


def dehaze(img_path):
    hazy_img = np.array(Image.open(img_path), dtype=np.float64)

    # create and save dark channel as JPEG image
    dark_ch = dehazing.dark_channel(hazy_img, 15)
    dark_ch_img = Image.fromarray(np.uint8(dark_ch), 'L')
    dark_ch_img.save(img_path.replace('.jpg', '_dark_channel.jpg'), format='JPEG')

    # compute atmospheric light from given hazy input image and corresponding dark channel
    atm_light = dehazing.atmospheric_light(hazy_img, dark_ch)

    # compute estimated transmission and save as JPEG
    transmission = dehazing.transmission(hazy_img, atm_light, 15)
    transmission_img = Image.fromarray(np.uint8(transmission * 255), 'L')
    transmission_img.save(img_path.replace('.jpg', '_est_transmission.jpg'), format='JPEG')

    # recover the scene radiance
    scene_radiance = dehazing.recover_radiance(hazy_img, atm_light, transmission)
    scene_radiance_img = Image.fromarray(np.uint8(scene_radiance), 'RGB')
    scene_radiance_img.save(img_path.replace('.jpg', '_scene_radiance.jpg'), format='JPEG')


if __name__ == '__main__':
    main()
