import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  
import numpy as np
import jax.numpy as jnp
import jax.random as jrnd
from functools import partial
import sys
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)
os.chdir(project_dir)
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Callable
from pyhocon import ConfigFactory
import argparse
from tqdm import tqdm
from orbax.checkpoint import CheckpointManagerOptions, CheckpointManager, PyTreeCheckpointer
import orbax.checkpoint as ocp
from flax.training import orbax_utils
# from flax import serialization
from models.plot_manager import PlotManager
import utils.general as utils
import utils.mesh_utils as mesh_utils


@dataclass
class CustomCheckpointMangerOptions(CheckpointManagerOptions):
    save_interval_steps: int = 1000

    # name of metric to use for checkpointing
    best_metric: Optional[str] = None


def save_checkpoint(checkpoint_manager, train_state, checkpoint_info, save_args, batch_index):
    # save_args=ocp.args.StandardSave(save_args)
    checkpoint_manager.save(
        step=batch_index,
        items={'model': train_state, **checkpoint_info},
        save_kwargs={'save_args': save_args},
        # args=ocp.args.StandardSave(save_args),
        force=True,
    )
    
def setup(args):

    modeldir = args.modeldir
    conf_path = os.path.join(modeldir,'runconf.conf')

    conf = ConfigFactory.parse_file(conf_path)

    conf.T = args.steps
    conf.expdir = modeldir
    
    utils.mkdir_ifnotexists(os.path.join(modeldir, 'eval'))
    plots_dir = os.path.abspath(os.path.join(modeldir, 'eval'))

    checkpoint_option = CustomCheckpointMangerOptions(**conf.check_point)
    
    checkpoint_manager = CheckpointManager(
        directory=os.path.join(modeldir,'checkpoints'),
        checkpointers=PyTreeCheckpointer(),
        options=checkpoint_option
    )
    
    plot_manager = PlotManager(
        directory=plots_dir,
        resolution=args.mc_resolution,
        f_batch_size=10000
        # **conf.plot
    )
    
    dataset = utils.get_class(conf.dataset_class)(**conf.datasets)

    return conf, checkpoint_manager, plot_manager, dataset


def eval_all(conf, checkpoint_manager, plot_manager, dataset):

    np.random.seed(conf.training.rng_seed)
    key = jrnd.PRNGKey(conf.training.rng_seed)
    learning_rate_fn = partial(
        utils.step_learning_rate_decay,
        initial=conf.training.initial,
        interval=conf.training.interval,
        factor=conf.training.factor)
    
    
    velocity_net = utils.get_class(conf.velocity_class)(num_shape=dataset.num_shape, **conf.network.velocity_net)
    implicit_net = utils.get_class(conf.implicit_class)(num_shape=dataset.num_shape, **conf.network.implicit_net)
    
    
    model = utils.get_class(conf.method)(T=conf.network.T, **conf.loss)

    key, = jrnd.split(key, 1)

    implicit_train_state, velocity_train_state = model.model_init(key, learning_rate_fn, implicit_net, velocity_net, conf)
    
    try:
        step = checkpoint_manager.latest_step()
        start_epoch = step
        # target = {'model':implicit_train_state, 'index': 0, 'pair': [0,1]}
        target = {'model':(implicit_train_state, velocity_train_state), 'start_frame': 0, 'length': 1}
        restored = checkpoint_manager.restore(step, items=target)
        implicit_train_state = restored['model'][0]
        velocity_train_state = restored['model'][1]
        
        start_frame = restored['start_frame']
        length = restored['length']
        
        
    
    except Exception as error:
        print('fail to load checkpoints.')
        
    dataset.preload(index=start_frame, length=length)
    
    implicit_fn = partial(implicit_train_state.apply_fn, implicit_train_state.params)
    velocity_fn = partial(velocity_train_state.apply_fn, velocity_train_state.params)
    
    for index in range(length):
        
        dptc_x, dptc_y, prefix = dataset.getitem(index)
        index_pair = dataset.get_index_pair(start_frame+index)
        
        bounding_box = mesh_utils.get_bounding_box(jnp.concatenate((dptc_x.points, dptc_y.points)))
                
        plot_manager(lower=bounding_box[0], upper=bounding_box[1], vertex_size=len(dptc_x.verts), prefix=prefix)
        
    
        # save gt shapes
        points = jnp.concatenate((dptc_x.verts, dptc_x.points))
        normals = jnp.concatenate((dptc_x.verts_normals, dptc_x.points_normals))
        plot_manager.save_pointcloud(points, normals,  output_file= '0_gt')
        points = jnp.concatenate((dptc_y.verts, dptc_y.points))
        normals = jnp.concatenate((dptc_y.verts_normals, dptc_y.points_normals))
        plot_manager.save_pointcloud(points, normals, output_file= '1_gt')
        
        if args.external:
            test_prefix = prefix.split('-')[1][:-1]
            external_points, external_colors = dataset.getitem_external_ptc(test_prefix)
            mesh = dataset.getitem_external_mesh(test_prefix)
            
            # save external gt target
            plot_manager.save_pointcloud(external_points, normals=None, color=external_colors,  output_file= 'external_ptc_1_gt')
            
            plot_manager.export_mesh(mesh, output_file='external_mesh_1_gt')
            
            test_prefix = prefix.split('-')[0]
            external_points, external_colors = dataset.getitem_external_ptc(test_prefix)
            mesh = dataset.getitem_external_mesh(test_prefix)
            verts_len = len(mesh.vertices)
            vertices = mesh.vertices
            vertex_colors = mesh.visual.vertex_colors

            try:
                faces = mesh.faces
            except:
                faces = None
            
            # save external gt source
            plot_manager.save_pointcloud(external_points, normals=None, color=external_colors,  output_file= 'external_ptc_0_gt')
            
            plot_manager.export_mesh(mesh, output_file='external_mesh_0_gt')
            
    
        # visualization
        points = jnp.concatenate((dptc_x.verts, dptc_x.points))
        color = plot_manager.get_color(points)
        
        feature=model.get_feature(velocity_train_state.params, index_pair)
        
        for time_step in range(args.steps + 1):
            t = time_step / args.steps
            print('point cloud time step {0}.....'.format(t))
            
            filename = 'time_' + str(time_step).zfill(2) + '_ptc'
            plot_manager.save_pointcloud(points=points, normals=None, color=color, output_file=filename)
            
            points = plot_manager.visualize_velocity(points, velocity_fn, time_step=t, index=index_pair)
            
            if not args.skip_recon:
                print('meshing time step {0}.....'.format(t))
                mesh = plot_manager.extract_mesh(implicit_fn, time_step=t, index=None, features=feature)
                filename = 'time_' + str(time_step).zfill(2) + '_mesh'
                plot_manager.export_mesh(mesh, filename)
            
            if args.external:
                filename = 'time_' + str(time_step).zfill(2) + '_external_ptc'
                plot_manager.save_pointcloud(points=external_points, normals=None, color=external_colors, output_file=filename)
                external_points = plot_manager.visualize_velocity(external_points, velocity_fn, time_step=t, index=index_pair)
                
                
                filename = 'time_' + str(time_step).zfill(2) + '_external_mesh'
                plot_manager.save_mesh(vertices, faces, color=vertex_colors, filename=filename)
                vertices = plot_manager.visualize_velocity(vertices, velocity_fn, time_step=t, index=index_pair)
            
    
    print('done.')


if __name__ == "__main__":
    
    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', type=str, default='./exp/fit_fraust/')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--mc_resolution',type=int, default=128)
    parser.add_argument('--skip_recon', action='store_true')
    parser.add_argument('--external', action='store_true')


    args = parser.parse_args()
    
    conf, checkpoint_manager, plot_manager, dataset = setup(
        args
    )

    eval_all(conf, checkpoint_manager, plot_manager, dataset)