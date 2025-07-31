import os, re
import numpy as np
import jax.numpy as jnp
import sys
from itertools import product
from glob import glob
import trimesh
sys.path.append('./Implicit-surf-Deformation')
from datasets.pointshape import DeformPointCloud
from natsort import natsorted
import jax.random as jrnd
import jax
from flax import serialization
from functools import partial


def sort_list(l):
    return natsorted(l)


class FlaxSequenceShape:
    
    def __init__(self, data_root, num_shape, index=0, length=1, batch_size=5000):
        
        assert os.path.isdir(data_root), f'Invaid data root.'
        
        self.flax_folder = os.path.join(data_root,'train')
        self.data_root = data_root
        
        self.index = index
        self.length =length
        
        # training pairs 
        self.flax_files = [f for f in os.listdir(self.flax_folder) if f.endswith('.flax')]
        self.flax_paths = sort_list(self.flax_files)
        
        # external test ptc
        self.test_folder = os.path.join(data_root, 'ptc')
        self.test_files = [f for f in os.listdir(self.test_folder) if f.endswith('.ply')]
        self.test_paths = sort_list(self.test_files)
        
        # external test mesh
        self.external = os.path.join(data_root,'mesh')
        self.external_files = [f for f in os.listdir(self.external) if f.endswith('.ply')]
        self.external_files = sort_list(self.external_files)
        
        
        self.flax_dptcs = []
        self.prefixs = []
        
        self.num_shape = num_shape
        self.batch_size = batch_size
        self.combinations = self.generate_index_pair()
        
        # self.preload()
                
                       
    def __len__(self):
        return len(self.flax_dptcs)
    
    
    def generate_index_pair(self):
        combination = [[i, i + 1] for i in range(len(self.flax_paths))]
        return combination

    def preload(self, index, length):
        dptc_list = self.generate_pesudo_dptc(20000)
        for i in range(index, index+length):
            
            filename = os.path.join(self.flax_folder, self.flax_paths[i])
            with open(filename,'rb') as f:
                bytes_data = f.read()
                
            loaded_instance = serialization.from_bytes(dptc_list, bytes_data)
            
            self.flax_dptcs.append(loaded_instance)
            self.prefixs.append(self.flax_paths[i][:-5]+"_")
    
        print('preloading is done, {0} pairs is loaded....'.format(len(self.flax_dptcs)))

    def getitem(self, index):
        
        if len(self.flax_dptcs) < index:
            raise ValueError('please preload pairs first....')
     
        dptc_list = self.flax_dptcs[index]
        dptc_x, dptc_y = dptc_list
    
        prefix = self.prefixs[index]
        
        return dptc_x, dptc_y, prefix
    
    
    def getitem_external_ptc(self, prefix):
        ptc_file = os.path.join(self.test_folder, prefix+'.ply')
        ptc = trimesh.load(ptc_file)
        points = ptc.vertices
        colors = ptc.visual.vertex_colors
        return points, colors
    
    
    def getitem_external_mesh(self, prefix):
        mesh = trimesh.load(os.path.join(self.external, prefix +'.ply'))
        
        return mesh
        
    
    
    def get_index_pair(self, index):
        return self.combinations[index]
    
    
    def generate_pesudo_dptc(self, point_num):
        dptc_x = DeformPointCloud(verts=jnp.zeros((5000,3)),
                verts_normals=jnp.zeros((5000,3)),
                points=jnp.zeros((point_num,3)),
                points_normals=jnp.zeros((point_num,3)), 
                local_sigma=jnp.zeros((point_num + 5000,3)),
                upper=jnp.ones((3)),
                lower=-jnp.ones((3))
        )  

        dptc_list = [dptc_x, dptc_x]
    
        return dptc_list


class FlaxPairShape:
    def __init__(self, data_root, num_shape,  batch_size=5000):
        
        assert os.path.isdir(data_root), f'Invaid data root.'
        
        self.flax_folder = os.path.join(data_root,'train')
        self.data_root = data_root
        
        # training pairs 
        self.flax_files = [f for f in os.listdir(self.flax_folder) if f.endswith('.flax')]
        self.flax_paths = sort_list(self.flax_files)
        
        # external test ptc
        self.test_folder = os.path.join(data_root, 'ptc')
        self.test_files = [f for f in os.listdir(self.test_folder) if f.endswith('.ply')]
        self.test_paths = sort_list(self.test_files)
        
        # external test mesh
        self.external = os.path.join(data_root,'mesh')
        self.external_files = [f for f in os.listdir(self.external) if f.endswith('.ply')]
        self.external_files = sort_list(self.external_files)
    
    
        self.flax_dptcs = []
        self.prefixs = []
        
        self.num_shape = num_shape
        self.batch_size = batch_size
        self.combinations = self.generate_index_pair()
        
    def __len__(self):
        return len(self.flax_dptcs)
    
    
    def preload(self, index, length):
        dptc_list = self.generate_pesudo_dptc(20000)
        for i in range(1, length+1):
            
            filename = os.path.join(self.flax_folder, self.external_files[index][:-4]+'-'+self.external_files[index+i][:-4]+'.flax')
            with open(filename,'rb') as f:
                bytes_data = f.read()
                
            loaded_instance = serialization.from_bytes(dptc_list, bytes_data)
            
            self.flax_dptcs.append(loaded_instance)
            self.prefixs.append(self.external_files[index][:-4]+'-'+self.external_files[index+i][:-4]+'_')
            
            print('load {0}.flax'.format(self.external_files[index][:-4]+'-'+self.external_files[index+i][:-4]))
        
        print('preloading is done, {0} pairs is loaded....'.format(len(self.flax_dptcs)))
    
    def generate_index_pair(self):
        shape_num = len(self.external_files)
        map_list = {}
        combination = []
        
        for index in range(shape_num):
            map_list.update({self.external_files[index][:-4]: index})
        
        for mesh_name in self.flax_paths:
            mesh_x = mesh_name.split('-')[0]
            mesh_y = mesh_name.split('-')[1][:-5]
            
            index1 = map_list[mesh_x]
            index2 = map_list[mesh_y]
            combination.append([index1, index2])
            
        return combination
    
    def get_internal_index(self, index, subindex):
        return self.combinations.index([index, subindex])
        
        
    def getitem(self, index):
        
        if len(self.flax_dptcs) < index:
            raise ValueError('please preload pairs first....')
     
        dptc_list = self.flax_dptcs[index]
        dptc_x, dptc_y = dptc_list
        prefix = self.prefixs[index]

        return dptc_x, dptc_y, prefix
    
    
    def get_index_pair(self, index):
        return self.combinations[index]
    
    
    def getitem_external_ptc(self, prefix):
        ptc_file = os.path.join(self.test_folder, prefix+'.ply')
        ptc = trimesh.load(ptc_file)
        points = ptc.vertices
        
        colors = ptc.colors
        return points, colors
    
    def getitem_external_mesh(self, prefix):
        mesh = trimesh.load(os.path.join(self.external, prefix +'.ply'))
        
        return mesh

    
    def generate_pesudo_dptc(self, point_num):
        dptc_x = DeformPointCloud(verts=jnp.zeros((5000,3)),
                verts_normals=jnp.zeros((5000,3)),
                points=jnp.zeros((point_num,3)),
                points_normals=jnp.zeros((point_num,3)), 
                local_sigma=jnp.zeros((point_num + 5000,3)),
                upper=jnp.ones((3)),
                lower=-jnp.ones((3)),
        )  

        dptc_list = [dptc_x, dptc_x]

        return dptc_list


if __name__ == "__main__":
    
    data_root = '/home/wiss/sang/git/4Deform/data/faust_r'
    num_shape = 20
    index = 0
    length = 5
    
    flax_seq = FlaxPairShape(data_root=data_root, num_shape=num_shape)
    
    dptc_list = flax_seq.preload(index, length)
    dptc_x, dptc_y = flax_seq.getitem(4)
    
    print('done')