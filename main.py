import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.functional import conv2d

from utils import *
from evaluation import CausalMetric, auc, gkern
from explanations import RISE

cudnn.benchmark = True

klen = 11
ksig = 5
kern = gkern(klen, ksig)

# Function that blurs input image
blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)

import torch
from einops.layers.torch import Reduce, Rearrange
import torchvision.transforms as transforms
import numpy as np
class BetterAGC:
    def __init__(self, model, attention_matrix_layer = 'before_softmax', attention_grad_layer = 'after_softmax', head_fusion='sum', layer_fusion='sum'):
        """
        Args:
            model (nn.Module): the Vision Transformer model to be explained
            attention_matrix_layer (str): the name of the layer to set a forward hook to get the self-attention matrices
            attention_grad_layer (str): the name of the layer to set a backward hook to get the gradients
            head_fusion (str): type of head-wise aggregation (default: 'sum')
            layer_fusion (str): type of layer-wise aggregation (default: 'sum')
        """
        self.model = model
        self.head = None
        self.width = None
        self.head_fusion = head_fusion
        self.layer_fusion = layer_fusion
        self.attn_matrix = []
        self.grad_attn = []

        for layer_num, (name, module) in enumerate(self.model.named_modules()):
            if attention_matrix_layer in name:
                module.register_forward_hook(self.get_attn_matrix)
            if attention_grad_layer in name:
                module.register_full_backward_hook(self.get_grad_attn)

    def get_attn_matrix(self, module, input, output):
        # As stated in Methodology part, in ViT with [class] token, only the first row of the attention matrix is directly connected with the MLP head.
        self.attn_matrix.append(output[:, :, 0:1, :]) # shape: [batch, num_heads, 1, num_patches]


    def get_grad_attn(self, module, grad_input, grad_output):
        # As stated in Methodology part, in ViT with [class] token, only the first row of the attention matrix is directly connected with the MLP head.
        self.grad_attn.append(grad_output[0][:, :, 0:1, :]) # shape: [batch, num_heads, 1, num_patches]


    def generate_cams_of_heads(self, input_tensor, cls_idx=None):
        self.attn_matrix = []
        self.grad_attn = []

        # backpropagate the model from the classification output
        self.model.zero_grad()
        output = self.model(input_tensor)
        _, prediction = torch.max(output, 1)
        self.prediction = prediction
        if cls_idx==None:                               # generate CAM for a certain class label
            loss = output[0, prediction[0]]
        else:                                           # generate CAM for the predicted class
            loss = output[0, cls_idx]
        loss.backward()

        b, h, n, d = self.attn_matrix[0].shape
        # b, h, n, d = self.attn_matrix.shape
        self.head=h
        self.width = int((d-1)**0.5)

        # put all matrices from each layer into one tensor
        self.attn_matrix.reverse()
        attn = self.attn_matrix[0]
        # attn = self.attn_matrix
        gradient = self.grad_attn[0]
        # gradient = self.grad_attn
        # layer_index = 2
        for i in range(1, len(self.attn_matrix)):
        # for i in range(layer_index, layer_index+1):
            # print('hia')
            attn = torch.concat((attn, self.attn_matrix[i]), dim=0)
            gradient = torch.concat((gradient, self.grad_attn[i]), dim=0)

        # As stated in Methodology, only positive gradients are used to reflect the positive contributions of each patch.
        # The self-attention score matrices are normalized with sigmoid and combined with the gradients.
        gradient = torch.nn.functional.relu(gradient) # Here, the variable gradient is the gradients alpha^{k,c}_h in Equation 7 in the methodology part.
        attn = torch.sigmoid(attn) # Here, the variable attn is the attention score matrices newly normalized with sigmoid, which are eqaul to the feature maps F^k_h in Equation 2 in the methodology part.
        mask = gradient * attn

        # aggregation of CAM of all heads and all layers and reshape the final CAM.
        mask = mask[:, :, :, 1:].unsqueeze(0) # * niên: chỗ này thêm 1 ở đầu (ví dụ: (2) -> (1, 2)) và 1: là bỏ token class
        # print(mask.shape)

        # *Niên:Thay vì tính tổng theo blocks và theo head như công thức để ra 1 mask cuối cùng là CAM thì niên sẽ giữ lại tất cả các mask của các head ở mỗi block
        mask = Rearrange('b l hd z (h w)  -> b l hd z h w', h=self.width, w=self.width)(mask) # *Niên: chỗ này tách từng token (1, 196) thành từng patch (1, 14, 14)

        return prediction, mask, output

    def generate_scores(self, head_cams, prediction, output_truth, image):
        with torch.no_grad():
            tensor_heatmaps = head_cams[0]
            tensor_heatmaps = tensor_heatmaps.reshape(144, 1, 14, 14)
            tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps)
    
            # Compute min and max along each image
            min_vals = tensor_heatmaps.amin(dim=(2, 3), keepdim=True)  # Min across width and height
            max_vals = tensor_heatmaps.amax(dim=(2, 3), keepdim=True)  # Max across width and height
            # Normalize using min-max scaling
            tensor_heatmaps = (tensor_heatmaps - min_vals) / (max_vals - min_vals + 1e-7)  # Add small value to avoid division by zero
            print("before multiply img with mask: ")
            print(torch.cuda.memory_allocated()/1024**2)
            m = torch.mul(tensor_heatmaps, image)
            print("After multiply img with mask scores: ")
            print(torch.cuda.memory_allocated()/1024**2)

            with torch.no_grad():
                output_mask = self.model(m)
            
            print("After get output from model: ")
            print(torch.cuda.memory_allocated()/1024**2)
    
            agc_scores = output_mask[:, prediction.item()] - output_truth[0, prediction.item()]
            agc_scores = torch.sigmoid(agc_scores)
    
            agc_scores = agc_scores.reshape(head_cams[0].shape[0], head_cams[0].shape[1])

            del output_mask  # Delete unnecessary variables that are no longer needed
            torch.cuda.empty_cache()  # Clean up cache if necessary
            print("After deleted output from model: ")
            print(torch.cuda.memory_allocated()/1024**2)
            
            return agc_scores

    def generate_saliency(self, head_cams, agc_scores):
        my_cam = (agc_scores.view(12, 12, 1, 1, 1) * head_cams[0]).sum(axis=(0, 1))
        mask = my_cam
        mask = mask.unsqueeze(0)
        # Reshape the mask to have the same size with the original input image (224 x 224)
        upsample = torch.nn.Upsample(224, mode = 'bilinear', align_corners=False)
        mask = upsample(mask)

        # Normalize the heatmap from 0 to 1
        mask = (mask-mask.min())/(mask.max()-mask.min())

        mask = mask.detach().cpu().numpy()[0]

        mask = np.transpose(mask, (1, 2, 0))

        return mask

    def __call__(self, x, class_idx=None):

        # Check that we get only one image
        assert x.dim() == 3 or (x.dim() == 4 and x.shape[0] == 1), "Only one image can be processed at a time"

        # Unsqueeze to get 4 dimensions if needed
        if x.dim() == 3:
            x = x.unsqueeze(dim=0)

        with torch.enable_grad():
            predicted_class, head_cams, output_truth = self.generate_cams_of_heads(x)

        print("After generate cams: ")
        print(torch.cuda.memory_allocated()/1024**2)
        print()
        
        # Define the class to explain. If not explicit, use the class predicted by the model
        if class_idx is None:
            class_idx = predicted_class
            print("class idx", class_idx)

        # Generate the saliency map for image x and class_idx
        scores = self.generate_scores(
            image=x,
            head_cams=head_cams,
            prediction=predicted_class, output_truth=output_truth
        )
        print("After generate scores: ")
        print(torch.cuda.memory_allocated()/1024**2)
        print()
        
        saliency_map = self.generate_saliency(head_cams=head_cams, agc_scores=scores)
        print("After generate saliency maps: ")
        print(torch.cuda.memory_allocated()/1024**2)
        print()

        return saliency_map

import PIL
# from better_agc.better_agc import BetterAGC
import Methods.AGCAM.ViT_for_AGCAM as ViT_Ours
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

MODEL = 'vit_base_patch16_224'
class_num = 1000
state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')

# explainer = RISE(model, (224, 224))
model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
model.load_state_dict(state_dict, strict=True)
model = model.eval()

from GPUtil import showUtilization as gpu_usage

print(gpu_usage())

explainer = BetterAGC(model)
# explainer.generate_masks(N=5000, s=10, p1=0.1)

# Image transform for ImageNet ILSVRC
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])
# For convert the input image into original distribution to display
unnormalize = transforms.Compose([
    transforms.Normalize([0., 0., 0.], [1/0.5, 1/0.5, 1/0.5]),
    transforms.Normalize([-0.5, -0.5, -0.5], [1., 1., 1.,])
])

IMAGE_PATH = '/kaggle/working/rise/goldfish.jpg'
image = PIL.Image.open(IMAGE_PATH)
image = transform(image)
image = image.unsqueeze(0).to('cuda')

sal = explainer(image)

print(gpu_usage())

from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter

from utils import *

HW = 224 * 224 # image area
n_classes = 1000

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, img_tensor, explanation, verbose=0, save_to=None):
        r"""Run metric on one image-saliency pair.

        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        
        pred = self.model(img_tensor)
        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1)
        for i in range(n_steps+1):
            
            pred = self.model(start)
            pr, cl = torch.topk(pred, 2)
            if verbose == 2:
                print('{}: {:.3f}'.format(get_class_name(cl[0][0]), float(pr[0][0])))
                print('{}: {:.3f}'.format(get_class_name(cl[0][1]), float(pr[0][1])))
            scores[i] = pred[0, c]
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                tensor_imshow(start[0])

                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start.cpu().numpy().reshape(1, 3, HW)[0, :, coords] = finish.cpu().numpy().reshape(1, 3, HW)[0, :, coords]
        return scores

    def evaluate(self, img_batch, exp_batch, batch_size):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, n_classes)
        assert n_samples % batch_size == 0
        for i in tqdm(range(n_samples // batch_size), desc='Predicting labels'):
            preds = self.model(img_batch[i*batch_size:(i+1)*batch_size]).cpu()
            predictions[i*batch_size:(i+1)*batch_size] = preds
        top = np.argmax(predictions, -1)
        n_steps = (HW + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.reshape(-1, HW), axis=1), axis=-1)
        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(img_batch)
        for j in tqdm(range(n_samples // batch_size), desc='Substrate'):
            substrate[j*batch_size:(j+1)*batch_size] = self.substrate_fn(img_batch[j*batch_size:(j+1)*batch_size])

        if self.mode == 'del':
            caption = 'Deleting  '
            start = img_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = img_batch.clone()

        # While not all pixels are changed
        for i in tqdm(range(n_steps+1), desc=caption + 'pixels'):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                
                preds = self.model(start[j*batch_size:(j+1)*batch_size])
                preds = preds.cpu().numpy()[range(batch_size), top[j*batch_size:(j+1)*batch_size]]
                scores[i, j*batch_size:(j+1)*batch_size] = preds
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords] = finish.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords]
        print('AUC: {}'.format(auc(scores.mean(1))))
        return scores

model = model.cpu()
insertion = CausalMetric(model, 'ins', 224, substrate_fn=blur)
deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)

gc.collect()
torch.cuda.empty_cache()
image = image.cpu()

h = insertion.single_run(image, sal, verbose=1)
