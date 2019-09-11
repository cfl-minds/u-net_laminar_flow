# Generic imports
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

# Custom imports
from params         import *
from datasets_utils import *
from networks_utils import *

# Load model
print('Loading model from disk')
model = load_model(model_json, model_h5)

# Load image dataset
imgs, n_imgs, height, width, n_channels = load_img_dataset(shape_dir_test,
                                                           downscaling,
                                                           'rgb')

# Load solutions dataset
sols, n_sols, height, width, n_channels = load_img_dataset(pressure_dir_test,
                                                           downscaling,
                                                           'rgb')

# Check consistency
if (n_imgs != n_sols):
    print('Error: I found',n_imgs,'image files and',n_sols,'solutions')
    quit(0)

# Make prediction and compute error
predicted_imgs, error, fail_count, max_err, mse = predict_images(model, imgs, sols)
pd.DataFrame(fail_count).to_csv('fail_count.csv')
plt.hist(fail_count, bins=20, density=False)
plt.savefig('number of failed pixels')
plt.show()
print(np.mean(fail_count))
#
pd.DataFrame(max_err).to_csv('max_err.csv')
plt.hist(max_err, bins=20, density=False)
plt.savefig('maximum absolute error')
plt.show()
print(np.mean(max_err))
##

pd.DataFrame(mse).to_csv('mse.csv')
plt.hist(mse, bins=20, density=False)
plt.savefig('mean-squared error')
plt.show()
print(np.mean(mse))

# Output a prediction example
for i in [351]:
     im = i
     show_image_prediction(sols[im, :, :, :],
                           predicted_imgs[im, :, :, :],
                           error[im, :, :], im)
#     ######intermedia output
#      subModel1 = Model(model.input, model.get_layer('conv2d_27').output)
#      subModel2 = Model(model.input, model.get_layer('conv2d_transpose_5').output)
#      subModel3 = Model(model.input, model.get_layer('zero_padding_1').output)
#      prediction1 = subModel1.predict(imgs[im, :, :, :].reshape(1, 78, 107, 3))[0, :, :, :]
#      prediction2 = subModel2.predict(imgs[im, :, :, :].reshape(1, 78, 107, 3))[0, :, :, :]
#      prediction3 = subModel3.predict(imgs[im, :, :, :].reshape(1, 78, 107, 3))[0, :, :, :]
# #
#
#
#
#     fig = plt.figure(figsize=(5, 15))
     # for j in range(8):
     #     for k in range(8):
     #         ax = fig.add_subplot(8, 8, 1+8*j+k)
     #         plt.imshow(prediction1[:, :, 8*j+k], cmap='gray')
     # plt.tight_layout()
     # plt.savefig('Intermediate output conv4')

#      ax = fig.add_subplot(2, 1, 1)
#      ax.set_title('Intermediate 1')
#      plt.imshow(prediction1)
#      ax = fig.add_subplot(2, 1, 2)
#      ax.set_title('Intermediate 2')
#      plt.imshow(prediction3)
# # #      ax = fig.add_subplot(3, 1, 3)
# # #      ax.set_title('normal U-net')
# # #      plt.imshow(prediction3)
#      plt.savefig('intermediate output {}'.format(i))
#      plt.show()

