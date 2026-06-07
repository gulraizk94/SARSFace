import os
import pickle
from os.path import join

import numpy as np
import trimesh
from util.facescape_bs import get_bs_weight, FaceScapeBlendshape
from copy import deepcopy
import logging


class OutputMeshBlendshapeKeyExp:
    def __init__(self, bilinear_model, predef_path):
        self.bilinear_model = bilinear_model
        self.template_mesh = trimesh.load(
            f'{predef_path}/convert_vt.obj',
            process=False)        
        self.full_template_mesh = trimesh.load(
            f'{predef_path}/convert_vt_full.obj',
            process=False, maintain_order=True,
            merge_tex=False,     # Forces isolation of vertices with different UVs
            merge_norm=False     # Forces isolation of vertices with different Normals
        )
        print(self.full_template_mesh.vertices.shape)
        self.indices = np.loadtxt(f'{predef_path}/front_vert_indices.txt', dtype=np.int32)
        self.bs_weight = get_bs_weight(full_bs=True)        

    def process(self, root_path, param=None):
        if param is None:
            files = os.listdir(root_path)
            if 'params.npz' in files:
                param = np.load(join(root_path, 'params.npz'))
            elif 'large_scale_param.pkl' in files:
                param = pickle.load(open(join(root_path, 'large_scale_param.pkl'), 'rb'))
            else:
                logging.warning(f'Cannot find param file in {root_path}')
                return

        id_mat = self.bilinear_model.get_id_mat(param['id'])
        verts = self.bilinear_model.get_posed_face_from_id_mat(id_mat, param['exp'])

        template_mesh = deepcopy(self.template_mesh)
        
        template_mesh.vertices[:] = verts[self.indices]
        trimesh.Trimesh(
            vertices=self.template_mesh.vertices,
            faces=self.template_mesh.faces
        ).export(join(root_path, 'current.obj'))
        
        if opt.show_mesh:
            
            face_mesh_visulize = trimesh.Trimesh( vertices=self.template_mesh.vertices, faces=self.template_mesh.faces)
            face_mesh_visulize.show()
    
    
        full_template_mesh = deepcopy(self.full_template_mesh)
        print(full_template_mesh.vertices.shape)
        full_template_mesh.vertices[:] = verts
        full_mesh = trimesh.Trimesh(
            vertices=verts,
            faces=self.full_template_mesh.faces,
            process=False
        )
        full_mesh.export(join(root_path, 'full_large.obj'))
         if opt.show_mesh:
            full_mesh.show()
        
        
        os.makedirs(join(root_path, 'bs'), exist_ok=True)
        for i in range(len(param['exp'])):
            new_exp = np.zeros(len(param['exp']))
            new_exp[0] = 1
            new_exp[i] = 1
            verts = self.bilinear_model.get_posed_face_from_id_mat(id_mat, new_exp)
            self.template_mesh.vertices[:] = verts[self.indices]
            mesh = trimesh.Trimesh(
                vertices=verts[self.indices],
                faces=self.template_mesh.faces,
                process=False
            )
            mesh.export(join(root_path, 'bs', f'{i}.obj'))
        os.makedirs(f'{root_path}/key_exp', exist_ok=True)
        for i, exp_name in enumerate(FaceScapeBlendshape.exp_list):
            print(str(i)+":"+exp_name)
            print(self.bs_weight[i])
            print(type(id_mat))
            np.save('facearray.npy', id_mat)


            self.bs_weight[i]=[0. , 0.  , 0. ,  0. ,  0.  , 0.  , 0.  , 0. ,  0. ,  0. ,  0.  , 0.  , 0.,   0.,
            0.  , 0. ,  0.  , 0. ,  0.  , 0., 0.  , 0.  , 0. ,  0. ,  0. ,  0. ,  0. ,  0.,
            0.8  ,0.6 , 0. ,  0. ,  0.  , 0. ,  0. ,  0.  , 0. ,  0. ,  0. ,  0. ,  0.  , 0.,
            0.  , 0.   ,0.  , 0.  , 0.  , 0.  , 0. ,  0.9 ,  0.  ]
            
            verts = self.bilinear_model.get_posed_face_from_id_mat(id_mat, np.concatenate(
                [np.ones(1), self.bs_weight[i]]))
            print(len(verts))
            template_mesh.vertices[:] = verts[self.indices]
            mesh = trimesh.Trimesh(
                vertices=template_mesh.vertices,
                faces=template_mesh.faces,
                process=False
            )
            mesh.export(f'{root_path}/key_exp/{exp_name}.obj')