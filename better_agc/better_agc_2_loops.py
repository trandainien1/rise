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

    def generate_scores(self, heatmaps, prediction, output, image):
        sum_heatmap = torch.tensor([]).to('cuda')
        #  Get each heatmap of each head in each layer
        for i in range(12):
            for j in range(12):
                
                heatmap = heatmaps[0][i][j]
    
                # Umpsampling the heatmap
                up_heatmap = heatmap.reshape(1, 1, 14, 14)
                upsample = torch.nn.Upsample(224, mode = 'bilinear', align_corners=False)
                up_heatmap = upsample(up_heatmap)
    
                # Normalize the heatmap
                norm_heatmap = (up_heatmap - up_heatmap.min())/(up_heatmap.max()-up_heatmap.min())
                # Generate new image
                new_image = image * norm_heatmap
    
                # Calculate the confidence and get the output using the new image
                masked_output = self.model.__predict__(new_image) 
                new_pred = torch.argmax(masked_output, dim = 1 )
    
                conf = masked_output - output
                conf = conf[0, prediction.item()].to('cuda') 
    
                # Generate new heatap
                sum_heatmap = torch.cat((sum_heatmap, conf.unsqueeze(0)), axis = 0)

            sigmoid_alpha = torch.sigmoid(sum_heatmap)
                
        return sigmoid_alpha

    def generate_saliency(self, heatmaps, sigmoid_alpha):
        sigmoid_alpha = sigmoid_alpha.unsqueeze(1).unsqueeze(2).repeat(1, 1, 196)
            
        sigmoid_heatmap = sigmoid_alpha * heatmaps.reshape(12 * 12, 1, 196)
        sigmoid_heatmap = torch.sum(sigmoid_heatmap, axis = 0)
    
        sigmoid_heatmap = sigmoid_heatmap.reshape(1, 1, 14, 14)
        sigmoid_heatmap = transforms.Resize((224, 224))(sigmoid_heatmap[0])
        sigmoid_heatmap = (sigmoid_heatmap - sigmoid_heatmap.min())/(sigmoid_heatmap.max()-sigmoid_heatmap.min())
        sigmoid_heatmap = sigmoid_heatmap.detach().cpu().numpy()
        sigmoid_heatmap = np.transpose(sigmoid_heatmap, (1, 2, 0))

        return sigmoid_heatmap

    def __call__(self, x, class_idx=None):

        # Check that we get only one image
        assert x.dim() == 3 or (x.dim() == 4 and x.shape[0] == 1), "Only one image can be processed at a time"

        # Unsqueeze to get 4 dimensions if needed
        if x.dim() == 3:
            x = x.unsqueeze(dim=0)

        with torch.enable_grad():
            predicted_class, head_cams, output = self.generate_cams_of_heads(x)

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
            heatmaps=head_cams,
            prediction=predicted_class, output=output
        )
        print("After generate scores: ")
        print(torch.cuda.memory_allocated()/1024**2)
        print()
        
        saliency_map = self.generate_saliency(heatmaps=head_cams, sigmoid_alpha=scores)
        print("After generate saliency maps: ")
        print(torch.cuda.memory_allocated()/1024**2)
        print()

        return saliency_map
