def AWGN_noise(img,filename):

  noise_path='/content/drive/MyDrive/AWGN_noise/'
  mean = 0
  std_dev = 25  
  noise = np.random.normal(mean, std_dev, img.shape)

  noisy_img = img + noise

  # Clip values to ensure they are in the valid pixel range [0, 255]
  noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
  cv2.imwrite(os.path.join(noise_path ,filename[:-4]+"_"+"awgn.png" ),noisy_img)
  