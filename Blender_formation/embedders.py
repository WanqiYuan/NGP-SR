import math
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class FullyConnectedLayer(nn.Module):
    def __init__(self,
        in_features,               
        out_features,               
        bias            = True,     
        activation      = 'linear', 
        lr_multiplier   = 1,        
        bias_init       = 0,        
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


def bias_act(x, b=None, act='linear'):
    if b is not None:
        x = x + b
    
    if act == 'relu':
        x = torch.relu(x)
    elif act == 'lrelu':
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
    elif act == 'sigmoid':
        x = torch.sigmoid(x)
    elif act == 'tanh':
        x = torch.tanh(x)
    
    return x


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def hash(coords, log2_hashmap_size=20):
    
    primes = torch.tensor([
        1, 11614769, 2654435761, 805459861, 2246822519, 3266489917, 
        4294967291, 1500450271, 1459629363, 273326509,  12341647, 13363367,
        15208767, 16430317, 17425307, 20394401, 21785619, 17249767,
        22039621, 23488747, 23879561, 24354221, 25157537, 25928687,
        26685441, 27144023, 27560431, 28074179, 28957989, 29674693,
        30424371, 31207067
    ][:coords.size(-1)], device=coords.device)    
    # primes = torch.tensor([1, 2654435761, 805459861, 2246822519, 3266489917, 4294967291][:coords.size(-1)], device=coords.device)
    # xor_result = torch.zeros(coords.size()[:-1], device=coords.device)
    xor_result = torch.zeros_like(coords)[..., 0]
    
    for i in range(coords.size(-1)):
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result


class BestHashPicker(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(BestHashPicker, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  
        )

    def forward(self, primes):
        return self.network(primes.float())
    

class CameraAwareInterpolator(nn.Module):
    def __init__(self,
        input_dim,                    
        camera_pe_size,               
        hidden_dim,                   
        output_dim,                  
        embed_features  = None,       
        num_layers      = 3,          
        activation      = 'tanh',     
        lr_multiplier   = 1,          
        normalize_input = True,       
    ):
        super().__init__()
        self.input_dim = input_dim
        self.camera_pe_size = camera_pe_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.normalize_input = normalize_input
        self.activation = activation
        
        if embed_features is None:
            embed_features = hidden_dim
        self.embed_features = embed_features
        
        
        self.camera_embed = FullyConnectedLayer(camera_pe_size, embed_features, 
                                               activation='tanh', lr_multiplier=lr_multiplier)

       
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()

        
        self.norm_layers = nn.ModuleList()
        
        
        if num_layers == 1:
            features_list = [input_dim + embed_features, output_dim]
        else:
            features_list = [input_dim + embed_features] + [hidden_dim + embed_features] * (num_layers - 2) + [hidden_dim + embed_features, output_dim]
            
            for _ in range(num_layers - 1):
                self.norm_layers.append(nn.LayerNorm(hidden_dim))
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_features = features_list[i]
            out_features = hidden_dim if i < num_layers - 1 else output_dim
            layer = FullyConnectedLayer(
                in_features, 
                out_features, 
                activation='tanh',  
                lr_multiplier=lr_multiplier
            )
            self.layers.append(layer)

    def forward(self, x, camera_pe):
        batch_size, num_points, input_dim = x.shape
        
        x_flat = x.reshape(batch_size * num_points, input_dim)
        
        if self.normalize_input:
            x_flat = normalize_2nd_moment(x_flat)
        
        camera_embed = self.camera_embed(camera_pe)
        
        if self.normalize_input:
            camera_embed = normalize_2nd_moment(camera_embed)
            
        camera_embed_expanded = camera_embed.unsqueeze(1).expand(-1, num_points, -1)
        camera_embed_flat = camera_embed_expanded.reshape(batch_size * num_points, -1)
        
        features = torch.cat([x_flat, camera_embed_flat], dim=1)
        
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                features = layer(features)
            else:
                hidden = layer(features)
                hidden = self.act(hidden)
                
                if i < len(self.norm_layers):
                    hidden = self.norm_layers[i](hidden)
                
                features = torch.cat([hidden, camera_embed_flat], dim=1)
        
        if self.activation == 'tanh':
            output = torch.tanh(features)
        else:
            output = torch.sigmoid(features)
        
        output = output.reshape(batch_size, num_points, self.output_dim)
        
        return output

    def extra_repr(self):
        return f'input_dim={self.input_dim:d}, camera_pe_size={self.camera_pe_size:d}, embed_features={self.embed_features:d}, hidden_dim={self.hidden_dim:d}, output_dim={self.output_dim:d}, num_layers={self.num_layers:d}, activation={self.activation:s}'


class HashEmbedder(nn.Module):
    def __init__(self, resolutions_list, bounding_box, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=512, mode='NN',
                 num_mf_layers=4):  
        super(HashEmbedder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.num_mf_layers = num_mf_layers  
        self.out_dim = num_mf_layers * self.n_features_per_level  
        self.n_dim = bounding_box[0].size(0)
        self.mode = mode

        self.resolutions = resolutions_list 

        self.mf_layer_indices = self._get_mf_layer_indices(num_mf_layers)
        self.mf_resolutions = {}
        self.mf_embeddings = nn.ModuleDict()
        
        for idx, layer_idx in enumerate(self.mf_layer_indices):
            key = f'MF{idx}'  
            resolution = int(self.resolutions[layer_idx])
            self.mf_resolutions[key] = resolution
            self.mf_embeddings[key] = nn.Embedding(2**self.log2_hashmap_size, self.n_features_per_level)
            nn.init.uniform_(self.mf_embeddings[key].weight, a=-0.0001, b=0.0001)

        if mode == 'interp':
            self.create_weight = CameraAwareInterpolator(
                input_dim=3 * self.n_dim,
                camera_pe_size=16,
                hidden_dim=64,
                output_dim=2**self.n_dim,
                embed_features=32,
                num_layers=3,
                activation='tanh',
                lr_multiplier=1
            )
        elif mode == 'NN':
            self.create_weight = CameraAwareInterpolator(
                input_dim=2**self.n_dim * n_features_per_level,
                camera_pe_size=16,
                hidden_dim=64,
                output_dim=2**self.n_dim,
                embed_features=32,
                num_layers=3,
                activation='tanh',
                lr_multiplier=1
            )

        self.primes = torch.tensor([
            1, 11614769, 2654435761, 805459861, 2246822519, 3266489917,
            4294967291, 1500450271, 1459629363, 273326509,  12341647, 13363367,
            15208767, 16430317, 17425307, 20394401, 21785619, 17249767,
            22039621, 23488747, 23879561, 24354221, 25157537, 25928687,
            26685441, 27144023, 27560431, 28074179, 28957989, 29674693,
            30424371, 31207067
        ], device=self.device)
        self.hash_picker = BestHashPicker(input_dim=len(self.primes), hidden_dim=64, output_dim=len(self.primes)).to(self.device)

    def _get_mf_layer_indices(self, num_mf_layers):
        print("number of mf")
        print(num_mf_layers)
        if num_mf_layers == 1:
            return [11]  
        elif num_mf_layers == 2:
            return [5, 11]  
        elif num_mf_layers == 3:
            return [3, 7, 11]  
        elif num_mf_layers == 4:
            return [2, 5, 8, 11]  
        elif num_mf_layers == 6:
            return [1, 3, 5, 7, 9, 11] 
        elif num_mf_layers == 12:
            return list(range(12)) 
        else:
            raise ValueError(f"Unsupported num_mf_layers: {num_mf_layers}. "
                           f"Supported values are: 1, 2, 3, 4, 6, 12")

    def pick_mf_key(self, i: int) -> str:
        for mf_idx, layer_idx in enumerate(self.mf_layer_indices):
            if i <= layer_idx:
                return f'MF{mf_idx}'
        return f'MF{len(self.mf_layer_indices) - 1}'

    def map_to_mf_coordinates(self, coords, original_resolution, mf_resolution):
        scale = mf_resolution / original_resolution
        return torch.floor(coords * scale)

    def n_linear_interp(self, voxel_embedds, x, voxel_min_vertex, voxel_max_vertex, camera_pe):
        B, N, NN, F = voxel_embedds.shape
        relative_min_dist = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex + 1e-12)
        relative_max_dist = (voxel_max_vertex - x) / (voxel_max_vertex - voxel_min_vertex + 1e-12)

        combined_input = torch.cat([x, relative_min_dist, relative_max_dist], dim=-1)  # [B, N, 3*n_dim]

        if self.mode == 'NN':
            weights = self.create_weight(voxel_embedds.reshape(B * N, -1), camera_pe)
            weights = weights.reshape(B, N, -1).unsqueeze(-1)
        else:  
            weights = self.create_weight(combined_input, camera_pe).unsqueeze(-1)

        weighted = voxel_embedds * weights          
        combined = torch.sum(weighted, dim=2)        
        return combined

    def get_voxel_vertices(self, xyz, bounding_box, resolution, log2_hashmap_size):
        device = xyz.device
        D = xyz.size(-1)
        box_min = bounding_box[0].to(device)
        box_max = bounding_box[1].to(device)

        xyz = torch.max(torch.min(xyz, box_max[None, None, :]), box_min[None, None, :])

        grid_size = (box_max - box_min) / resolution
        bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
        voxel_min_vertex = bottom_left_idx * grid_size + box_min
        voxel_max_vertex = voxel_min_vertex + grid_size

        BOX_OFFSETS = torch.tensor(list(product([0, 1], repeat=D)), dtype=torch.int32, device=xyz.device)

        expanded_idx = bottom_left_idx.unsqueeze(2) 
        total_offsets = BOX_OFFSETS.unsqueeze(0).unsqueeze(0)  
        voxel_indices = expanded_idx + total_offsets  

        flat_indices = voxel_indices.view(-1, D)
        hashed_voxel_indices = hash(flat_indices, log2_hashmap_size).view(xyz.shape[0], xyz.shape[1], -1)

        return xyz, voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

    def get_voxel_vertices_coordinates_only(self, xyz, bounding_box, resolution, log2_hashmap_size):
        device = xyz.device
        D = xyz.size(-1)
        box_min = bounding_box[0].to(device)
        box_max = bounding_box[1].to(device)

        xyz = torch.max(torch.min(xyz, box_max[None, None, :]), box_min[None, None, :])

        grid_size = (box_max - box_min) / resolution
        bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
        voxel_min_vertex = bottom_left_idx * grid_size + box_min
        voxel_max_vertex = voxel_min_vertex + grid_size

        empty_hash = torch.zeros(xyz.shape[0], xyz.shape[1], 2**D, device=device, dtype=torch.long)
        return xyz, voxel_min_vertex, voxel_max_vertex, empty_hash

    def forward(self, x, i, camera_pe):
        original_resolution = int(self.resolutions[i])
        x_lvl, vmin_lvl, vmax_lvl, _ = self.get_voxel_vertices_coordinates_only(
            x, self.bounding_box, original_resolution, self.log2_hashmap_size
        )

        mf_key = self.pick_mf_key(i)                   
        mf_resolution = self.mf_resolutions[mf_key]    

        x_mf, vmin_mf, vmax_mf, hashed_idx_mf = self.get_voxel_vertices(
            x_lvl, self.bounding_box, mf_resolution, self.log2_hashmap_size
        )

        voxel_embedds = self.mf_embeddings[mf_key](hashed_idx_mf.to(self.device)) 
        x_embedded = self.n_linear_interp(voxel_embedds, x_mf, vmin_mf, vmax_mf, camera_pe) 

        return x_embedded, vmin_mf

    def get_mf_info(self):
        info = {
            'num_mf_layers': self.num_mf_layers,
            'mf_layer_indices': self.mf_layer_indices,
            'mf_resolutions': self.mf_resolutions,
            'out_dim': self.out_dim
        }
        return info


class CameraPatchMLP(nn.Module):
    def __init__(self,
        patch_size,                
        camera_pe_size,            
        hidden_size,              
        output_size,               
        embed_features = None,     
        num_layers     = 2,        
        activation     = 'tanh',   
        lr_multiplier  = 1,        
        normalize_input = True,   
    ):
        super().__init__()
        self.patch_size = patch_size
        self.camera_pe_size = camera_pe_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.normalize_input = normalize_input
        self.activation = activation
        
        if embed_features is None:
            embed_features = hidden_size
        self.embed_features = embed_features
        
        self.camera_embed = FullyConnectedLayer(
            camera_pe_size, 
            embed_features, 
            activation='tanh', 
            lr_multiplier=lr_multiplier
        )
        
        self.norm_layers = nn.ModuleList()
        
        if num_layers == 1:
            features_list = [patch_size + embed_features, output_size]
        else:
            features_list = [patch_size + embed_features] + [hidden_size + embed_features] * (num_layers - 2) + [hidden_size + embed_features, output_size]
            
            for _ in range(num_layers - 1):
                self.norm_layers.append(nn.LayerNorm(hidden_size))
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_features = features_list[i]
            out_features = hidden_size if i < num_layers - 1 else output_size
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation='tanh', 
                lr_multiplier=lr_multiplier
            )
            self.layers.append(layer)
        
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()

    def forward(self, patches, camera_pe):
        B, num_patches, patch_size = patches.shape
        
        patches_flat = patches.reshape(B * num_patches, patch_size)
        
        if self.normalize_input:
            patches_flat = normalize_2nd_moment(patches_flat)
        
        camera_embed = self.camera_embed(camera_pe)
        
        if self.normalize_input:
            camera_embed = normalize_2nd_moment(camera_embed)
        
        camera_embed_expanded = camera_embed.unsqueeze(1).expand(-1, num_patches, -1)
        camera_embed_flat = camera_embed_expanded.reshape(B * num_patches, -1)
        
        features = torch.cat([patches_flat, camera_embed_flat], dim=1)
        
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                features = layer(features)
            else:
                hidden = layer(features)
                hidden = self.act(hidden)
                
                if i < len(self.norm_layers):
                    hidden = self.norm_layers[i](hidden)
                
                features = torch.cat([hidden, camera_embed_flat], dim=1)
        
        if self.activation == 'tanh':
            output = torch.tanh(features)
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(features)
        else:
            output = features
        
        output = output.reshape(B, num_patches, self.output_size)
        
        return output

    def extra_repr(self):
        return f'patch_size={self.patch_size:d}, camera_pe_size={self.camera_pe_size:d}, embed_features={self.embed_features:d}, hidden_size={self.hidden_size:d}, output_size={self.output_size:d}, num_layers={self.num_layers:d}, activation={self.activation:s}'
    

class PatchEmbedder(nn.Module):
    def __init__(self, image_size, resolutions_list, n_patch_feature, bounding_box, n_levels=16, base_resolution=16, finest_resolution=512,activation='tanh'):
        super(PatchEmbedder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        C, H, W = image_size
        
        self.C, self.H, self.W = C, H, W
        self.height = finest_resolution
        self.width = finest_resolution
        self.n_pixel = H*W
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.b = torch.exp((torch.log(torch.tensor(finest_resolution, dtype=torch.float32)) - 
                            torch.log(torch.tensor(base_resolution, dtype=torch.float32))) / (n_levels - 1)).to(self.device)
        

        self.resolutions = resolutions_list
    
        self.mlps = nn.ModuleList()
        self.patch_heights = []
        self.patch_widths = []
        for i in range(n_levels):
            resolution = self.resolutions[i]
            patch_height = finest_resolution // resolution
            patch_width = finest_resolution // resolution
            patch_size = C * patch_height * patch_width
            
            mlp = CameraPatchMLP(
                patch_size=patch_size,
                camera_pe_size=16,
                hidden_size=128,
                output_size=n_patch_feature,
                embed_features=64,      
                num_layers=2,          
                activation='tanh',      
                lr_multiplier=1,        
            )
            self.mlps.append(mlp.to(self.device))  
        

        self.activation = activation
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, steps=H),
            torch.linspace(0, 1, steps=W)
        )            
        coords = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2)
        self.array_of_coords = coords.to(self.device)

    def forward(self, patches_list, patch_indices):
        results = []
        B = patch_indices[0].shape[0] 
   
        camera_matrices = patches_list[0]['camera_matrix'] 
        #print("Camera matrices shape:", camera_matrices.shape)
        #print("camera_matrices_size",camera_matrices.shape)
        camera_pe = camera_matrices.reshape(camera_matrices.shape[0], -1) 
        #print("camera_pe.shape",camera_pe.shape)
        

        for i, patch_encoder in enumerate(self.mlps):
            patches = patches_list[i]['patches']
            patch_index = patch_indices[i]
            #print("patchese.shape",patches.shape)
            #print("camera_pe.shape",camera_pe.shape)

            feature_vectors = patch_encoder(patches, camera_pe)
        
            if self.activation == 'tanh':
                feature_vectors = torch.tanh(feature_vectors)
            else:
                feature_vectors = torch.sigmoid(feature_vectors)
            
            selected_features = feature_vectors[torch.arange(B)[:, None], patch_index]
            
            #print("feature_vectors",feature_vectors)
            
            results.append(selected_features)

        results_tensor = torch.stack(results, dim=0)
        
        return results_tensor, camera_pe
        #print("results_tensor",results_tensor)
    

class EfficientCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=4, dim_head=None, dropout=0.0):
        super().__init__()
        self.heads = heads
        head_dim = dim_head if dim_head is not None else query_dim // heads
        inner_dim = head_dim * heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        
        self.dropout_p = dropout
        
    def forward(self, x, context=None):
        context = context if context is not None else x
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        batch_size, query_len, _ = q.shape
        context_len = k.shape[1]
        
        q = q.reshape(batch_size, query_len, self.heads, -1).transpose(1, 2)
        k = k.reshape(batch_size, context_len, self.heads, -1).transpose(1, 2)
        v = v.reshape(batch_size, context_len, self.heads, -1).transpose(1, 2)
        
        dropout_p = self.dropout_p if self.training else 0.0
        try:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except AttributeError:
            scale = (q.shape[-1]) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            if dropout_p > 0:
                attn = F.dropout(attn, p=dropout_p)
            out = attn @ v
            
        out = out.transpose(1, 2).reshape(batch_size, query_len, -1)
        return self.to_out(out)
    

class EnhancedFeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, output_channels=32, hidden_channels=64):
        super(EnhancedFeatureExtractor, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1)
        
        self.skip_connection = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        
        self.spatial_attention = LightweightSpatialAttention()
        
        self.relu = nn.ReLU()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(output_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, output_channels)
        )
        
        self.feature_weighting = nn.Sequential(
            nn.Linear(output_channels, output_channels),
            nn.Sigmoid()
        )
    
    def forward(self, lr_img):
        B, N, C = lr_img.shape
        H = W = int(math.sqrt(N))
        
        img = lr_img.permute(0, 2, 1).reshape(B, C, H, W)
        
        residual = self.skip_connection(img)
        
        x = self.relu(self.conv1(img))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        
        x = x + residual
        x = self.relu(x)
        
        x = self.spatial_attention(x)
        
        global_features = self.global_pool(x).squeeze(-1).squeeze(-1) 
        global_features = self.feature_mlp(global_features) 
        
        pixel_features = x.reshape(B, -1, N).permute(0, 2, 1) 
        
        global_features_expanded = global_features.unsqueeze(1).expand(-1, N, -1) 
        
        weights = self.feature_weighting(global_features_expanded)
        
        combined_features = pixel_features * weights + global_features_expanded * (1 - weights)
        
        return combined_features


class LightweightSpatialAttention(nn.Module):
    def __init__(self):
        super(LightweightSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        attention = torch.cat([avg_out, max_out], dim=1)
        
        attention = self.conv(attention)
        
        attention = torch.sigmoid(attention)
        
        return x * attention
    

class CustomMappingNetwork(nn.Module):
    def __init__(self,
        z_dim,                     
        c_dim,                      
        w_dim,                     
        num_layers      = 8,        
        embed_features  = None,    
        layer_features  = None,     
        activation      = 'tanh',  
        lr_multiplier   = 0.01,    
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_layers = num_layers
        
        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
            
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        
        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
            
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)
    
    def forward(self, z, c):
        if len(z.shape) > 2:
            batch_size = z.shape[0] * z.shape[1]
            z = z.reshape(batch_size, -1)
        
        if len(c.shape) > 2:
            c = c.reshape(-1, self.c_dim)
        
        x = normalize_2nd_moment(z.to(torch.float32))
        
        if self.c_dim > 0:
            y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
            x = torch.cat([x, y], dim=1)
            
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)
            
        return x


class TwoStageMappingFusion(nn.Module):
    def __init__(self, 
                 feature_dim,      
                 camera_pe_dim,    
                 lr_img_dim=3,    
                 hidden_dim=64,    
                 num_layers=4):    
        super().__init__()
        self.feature_dim = feature_dim
        self.camera_pe_dim = camera_pe_dim
        self.lr_img_dim = lr_img_dim
        
        self.img_feature_dim = 32  
        
        self.camera_mapping = CustomMappingNetwork(
            z_dim=feature_dim,
            c_dim=camera_pe_dim,
            w_dim=hidden_dim,
            num_layers=num_layers,
            embed_features=hidden_dim//2,
            activation='tanh'
        )
        
        self.img_mapping = CustomMappingNetwork(
            z_dim=hidden_dim,
            c_dim=self.img_feature_dim,
            w_dim=feature_dim,
            num_layers=num_layers,
            embed_features=hidden_dim//2,
            activation='tanh'
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x, camera_pe, lr_img, enhanced_feature_extractor):
        B, N, _ = x.shape
        
        x_flat = x.reshape(B*N, -1)
        
        camera_pe_expanded = camera_pe.unsqueeze(1).expand(-1, N, -1).reshape(B*N, -1)
        
        A = self.camera_mapping(x_flat, camera_pe_expanded)
        
        image_features = enhanced_feature_extractor(lr_img)
        
        B_out = self.img_mapping(A, image_features)
        
        output = B_out.reshape(B, N, self.feature_dim)
        
        return output
    

class Camera_LR_PreNet(nn.Module):
    def __init__(self, input_dim, camera_pe_size, hidden_dim, output_dim):
        super(Camera_LR_PreNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, camera_pe=None):
        B, N, input_dim = x.shape
        
        x_flat = x.reshape(B*N, -1)
        x1 = self.layer1(x_flat)
        x1 = self.relu(x1)
        x1 = x1.reshape(B, N, -1)
        x1 = self.norm1(x1)
        
        x1_flat = x1.reshape(B*N, -1)
        x2 = self.layer2(x1_flat)
        x2 = self.relu(x2)
        x2 = x2.reshape(B, N, -1)
        x2 = self.norm2(x2)
        
        x2_flat = x2.reshape(B*N, -1)
        x3 = self.layer3(x2_flat)
        x3 = self.relu(x3)
        x3 = x3.reshape(B, N, -1)
        x3 = self.norm3(x3)
        
        x3_flat = x3.reshape(B*N, -1)
        x4 = self.layer4(x3_flat)
        x4 = self.relu(x4)
        x4 = x4.reshape(B, N, -1)
        x4 = self.norm4(x4)
        
        x4_flat = x4.reshape(B*N, -1)
        x5 = self.layer5(x4_flat)
        
        output = x5.reshape(B, N, -1)
        
        return output
    

class AttentionMLP(nn.Module):
    def __init__(self, input_dim):
        super(AttentionMLP, self).__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),   
            # nn.Linear(64, 64),
            # nn.ReLU(),              
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.attention_mlp(x)

class CameraAwareNetwork(nn.Module):
    def __init__(self, input_dim, camera_pe_size, hidden_dim, output_dim):
        super(CameraAwareNetwork, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        B, N, input_dim = x.shape
        
        x_flat = x.reshape(B*N, -1)
        
        x1 = self.layer1(x_flat)
        x1 = self.relu(x1)
        x1 = x1.reshape(B, N, -1)
        x1 = self.norm1(x1)
        
        x1_flat = x1.reshape(B*N, -1)
        x2 = self.layer2(x1_flat)
        x2 = self.relu(x2)
        x2 = x2.reshape(B, N, -1)
        x2 = self.norm2(x2)
        
        x2_flat = x2.reshape(B*N, -1)
        x3 = self.layer3(x2_flat)
        
        output = torch.sigmoid(x3).reshape(B, N, -1)
        
        return output


