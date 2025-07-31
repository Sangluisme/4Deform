import jax
import flax
from flax import linen as nn
from jax import random, jit, vmap
from functools import partial
import jax.random as jrnd
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from models.utils import *
from datasets import sampler
from datasets.pointshape import DeformPointCloud


@flax.struct.dataclass
class Deform:
    alpha: float = 0.05
    reinitial: float = 0
    laplacian: float = 10
    divergence: float = 10
    level_set: float = 500
    normal: float = 100
    match: float = 200
    manifold: float = 100
    nonmanifold: float = 10
    eikonal: float = 10
    surf_area: float=10
    stretching: float = 5
    distortion: float = 5
    bending: float = 5
    T: int= 8
 
    
    
    def model_init(self, key, learning_rate_fn, implicit_net, velocity_net, conf):
        key_i, key_v = jrnd.split(key, 2)
        if conf.implicit_class.split('.')[-1] == 'Decoder':
            implicit_train_state = Decoder_init(key_i, learning_rate_fn, implicit_net, conf.network.implicit_net)
        elif conf.implicit_class.split('.')[-1] == 'MLP':
            implicit_train_state = MLP_init(key_i, learning_rate_fn, implicit_net, conf.network.implicit_net)
        else:
            raise NotImplementedError
        
        if conf.velocity_class.split('.')[-1] == 'Decoder':
            velocity_train_state = Decoder_init(key_v, learning_rate_fn, velocity_net, conf.network.velocity_net)
        elif conf.velocity_class.split('.')[-1] == 'MLP':
            velocity_train_state = MLP_init(key_v, learning_rate_fn, velocity_net, conf.network.velocity_net)
        else:
            raise NotImplementedError

        return implicit_train_state, velocity_train_state
    
    def get_ratio(self, epoch, start_epoch, nepoch):
        
        end = np.min([nepoch // 2, start_epoch *3])

        if epoch < start_epoch:
            ratio_v = 1.0
            ratio_i = 0.0
            
        elif epoch < end:
            ratio_i = np.min([1.0, (epoch - start_epoch) / (end - start_epoch)])
            ratio_v = 1.0
            
        else:
            ratio_v = 0.0
            ratio_i = 1.0
        
        return [ratio_v, ratio_i]
    
    def visual(self, plot_manager, dptc, index, feature=None, velocity_fn=None, implicit_fn=None, epoch=0, ratio=()):
            ratio_v, ratio_i = ratio
            points = jnp.concatenate((dptc.verts, dptc.points))
            
            if ratio_i > 0:
                self.visual_both(plot_manager, points, index=index, feature=feature, velocity_fn=velocity_fn, implicit_fn=implicit_fn, epoch=epoch)
            else:
                self.visual_velocity(plot_manager, points, index=index, feature=feature, velocity_fn=velocity_fn, epoch=epoch)
                
                
    def visual_both(self, plot_manager, points, index, feature=None, velocity_fn=None, implicit_fn=None, epoch=0):
        
        
        color = plot_manager.get_color(points)
        
        for time in range(self.T+1):
            
            t = time / self.T
           
            print('point cloud time step {0}.....'.format(t))
            filename = 'epoch_' + str(epoch) + '_time_' + str(time) + '_ptc'
            plot_manager.save_pointcloud(points=points, normals=None, color=color, output_file=filename)

            points = plot_manager.visualize_velocity(points, velocity_fn, time_step=t, index=index)
            
            print('meshing time step {0}.....'.format(t))
            mesh = plot_manager.extract_mesh(implicit_fn, time_step=t, index=None, features=feature)
            filename = 'epoch_' + str(epoch) + '_time_' + str(time) + '_mesh'
            plot_manager.export_mesh(mesh, filename)
    
    
    def visual_velocity(self, plot_manager, points, index, feature=None, velocity_fn=None, epoch=0):
        color = plot_manager.get_color(points)
        
        for time in range(self.T+1):
           
            print('point cloud time step {0}.....'.format(time))
            filename = 'epoch_' + str(epoch) + '_time_' + str(time) + '_ptc'
            plot_manager.save_pointcloud(points=points, normals=None, color=color, output_file=filename)

            points_slice = grid_slice(points, step=10000)
        
            output = []
            
            for p in points_slice:
                t = jnp.tile(time/ self.T,  (p.shape[0],1))
                velocity = velocity_fn(p, t=t, index=index)

                p_curr = p + velocity
                output.append(p_curr)
            
            points = jnp.concatenate(output)
            
    @partial(jit, static_argnames=("self", "batch_size"))
    def get_batch(self, key, batch_size, dptc_x: DeformPointCloud, dptc_y: DeformPointCloud):
        key_v, key_p1, key_p2, key_local, key_global = jrnd.split(key, 5)

        x, nx, indices = sampler.sample_array(key_v, batch_size, dptc_x.verts, dptc_x.verts_normals)
        y = dptc_y.verts[indices]
        ny = dptc_y.verts_normals[indices]

        sx, snx, indices1 = sampler.sample_array(key_p1, batch_size, dptc_x.points, dptc_x.points_normals)
        # sy = dptc_y.points[indices]
        # sny = dptc_y.points_normals[indices]
        sy, sny, indices2 = sampler.sample_array(key_p2, batch_size, dptc_y.points, dptc_y.points_normals)

        sample_local_x = sampler.generate_local_samples(key_local, sx, dptc_x.local_sigma[indices1])

        sample_local_y = sampler.generate_local_samples(key_local, sy, dptc_y.local_sigma[indices2])

        sample_global = sampler.generate_global_samples(key_global, dptc_x.lower, dptc_x.upper, batch_size, 3)

        input_x = jnp.concatenate((x, sx))
        input_nx = jnp.concatenate((nx, snx))
        input_y = jnp.concatenate((y, sy)) 
        input_ny = jnp.concatenate((ny, sny))
        sample_local_x = jnp.concatenate((sample_local_x, sample_global))
        sample_local_y = jnp.concatenate((sample_local_y, sample_global))

        return input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global
    
    # only fit for MLP implicit
    def get_full_gradient(self, f, params, points, t, condition=None):
        
        sdf, (df, dt) = jax.value_and_grad(f, argnums=(1,2))(params, points, t, condition=condition)
            
        return sdf, df, dt
    
    # only get df
    def get_surface_gradient(self, f, params, points, t=None, condition=None):
        df = jax.jacobian(f,argnums=1)(params, points, t, condition=condition)
        return df
    
    
    def get_implicit_surface_area_loss(self, f, points, t, condition=None, alpha=100):    
        loss = jnp.exp(-alpha * jnp.abs(f(points, t=t, condition=condition))).mean()
        return loss 
    
    def get_surface_area_loss(self, sdf, alpha=100):
        return jnp.exp(-alpha * jnp.abs(sdf)).mean()

    
    def get_v_jacobian(self, f, params, points, t=None, index=None):
        if index is not None:
            dv = jax.jacobian(f, argnums=1)(params, points, t=t, index=index)
        else:
            dv = jax.jacobian(f, argnums=1)(params, points, t=t)
        return dv 
    
    def get_v_hessian(self, f, params, points, t, index=None):
        if index is not None:
            Hv = jax.hessian(f, argnums=1)(params, points, t=t, index=index)
        else:
            Hv = jax.hessian(f, argnums=1)(params, points, t=t)
        return Hv 
    
    def get_v_divergence(self, f, params, points, t, index=None):
        
        dv = vmap(self.get_v_jacobian, in_axes=(None, None, 0, 0, None))(f, params, points, t, index=index)
        
        div_v = jnp.trace(dv, axis1=1, axis2=2)
        return jnp.abs(div_v)
    

    def get_feature(self, params_v, index_pair):
        feature_x = params_v['params']['Embed_0']['embedding'][index_pair[0]]
        feature_y = params_v['params']['Embed_0']['embedding'][index_pair[1]]
        return jnp.concatenate((feature_x, feature_y), axis=-1)
    
    
    def get_loss(self, f_fn, params, fv_fn, params_v, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, index_pair, ratio):
        
        batch_size = input_x.shape[0] // 2
        
        points = input_x

        loss, loss_lse, loss_surf, loss_normal, loss_area = 0.0, 0.0, 0.0, 0.0, 0.0 #total loss, level-set loss, on surface loss, normal loss, minimum area loss
        
        loss_en, loss_es = 0.0, 0.0 #eikonal on non-surf, eikonal on surf
        
        loss_lap, loss_div, loss_match = 0.0, 0.0, 0.0 #smooth loss, divergence loss, matching loss
        
        loss_distort, loss_stretch, loss_bending = 0.0, 0.0, 0.0 # distortion loss, stretching loss, bending loss (on supplementary normal deformation, normally not used)
        
        ratio_v, ratio_i = ratio

        # fit first t==0:
        t = jnp.tile(0., (input_x.shape[0],1))
        df_last = input_nx
        
        features = self.get_feature(params_v=params_v, index_pair=index_pair)
        condition = jnp.tile(features, (points.shape[0], 1))
        
        for time in range(self.T+1):

            t = jnp.tile(time/self.T, (points.shape[0],1))


            sdf, df, dt = vmap(self.get_full_gradient, in_axes=(None, None, 0, 0, 0))(f_fn, params, points, t, condition)
            
            v = partial(fv_fn, params_v)(points, t, index_pair)
            dv = vmap(self.get_v_jacobian, in_axes=(None, None, 0, 0, None))(fv_fn, params_v, points, t, index_pair)

            if (self.eikonal > 0) and (self.reinitial == 0):
                # loss_e += ((1-jnp.linalg.norm(df, axis=-1))**2).mean()
                loss_es += ((1-soft_norm(df, axis=-1))**2).mean()
            
            if self.nonmanifold > 0:
                df_s = vmap(self.get_surface_gradient, in_axes=(None, None, 0, 0, 0))(f_fn, params, sample_local_x, t, condition)
                loss_en += ((1-soft_norm(df_s, axis=-1))**2).mean()
                v_sample = partial(fv_fn, params_v)(sample_local_x, t, index_pair)
                sample_local_x = sample_local_x + v_sample
                
            if self.surf_area > 0:
                loss_area += self.get_implicit_surface_area_loss(partial(f_fn, params), sample_global, t[:batch_size,:], condition[:batch_size,:])

            if self.level_set > 0:
                if self.reinitial > 0:
                    R = (self.eikonal/self.level_set) * R_term(df, dv)
                    loss_lse += sdf_loss(dt + jnp.sum(df * v, axis=-1) + sdf*R,'squared_l2')
                    
                else:
                    loss_lse += sdf_loss(dt + jnp.sum(df * v, axis=-1), 'squared_l2')
                    loss_lse += (self.eikonal/self.level_set) * ((1-soft_norm(df, axis=-1))**2).mean()

            if self.manifold > 0:
                loss_surf += sdf_loss(sdf, distance_metric='l2')
            
            if (self.manifold > 0):
                loss_surf += sdf_loss(sdf,'l2')
            
            # normal loss
            if (self.normal > 0) and (time == 0):
                loss_normal += normal_loss(input_nx, df)
            if (self.normal > 0) and (time==self.T):
                loss_normal += normal_loss(df, input_ny)
               
            if (self.laplacian > 0.0) and (time < self.T):
                Hv = vmap(self.get_v_hessian, in_axes=(None, None, 0, 0, None))(fv_fn, params_v, points, t, index_pair)
                laplacian = jnp.trace(Hv, axis1=2, axis2=3)
                # loss_lap += jnp.linalg.norm(v - self.alpha * laplacian, axis=-1).mean()
                loss_lap += sq_norm(v - self.alpha * laplacian, axis=-1).mean()                   

            if (self.divergence > 0.0) and (time < self.T):
                
                div_v = jnp.trace(dv, axis1=1, axis2=2)
                loss_div += sdf_loss(div_v,'squared_l2')
            
            if (self.match > 0) and (time==self.T):
                loss_match = match_loss(points[:batch_size,:], input_y[:batch_size,:])
            
            if self.distortion > 0:
                deformation = deformation_rate(dv)
                loss_distort += sdf_loss(deformation, 'squared_l2')
            
            if (self.bending > 0) and (time > 0):
                loss_bending += deformation_gradient(df_last, df, dv)

        
            if self.stretching > 0:
                loss_stretch += stretching_loss(df, dv) 
            
            df_last = df
            
            # move points
            points = points + v
            
        # fit second
        t = jnp.tile(1.0, (input_y.shape[0],1))
        
        sdf_y, df_y, dt = vmap(self.get_full_gradient, in_axes=(None, None, 0, 0, 0))(f_fn, params, input_y, t, condition)
        
        if self.manifold > 0:
            loss_surf += sdf_loss(sdf_y, distance_metric='l2')
            
        if self.nonmanifold > 0:
            sdf_s, df_s, dt_s = vmap(self.get_full_gradient, in_axes=(None, None, 0, 0, 0))(f_fn, params, sample_local_y, t, condition)
            loss_en += ((1-soft_norm(df_s, axis=-1))**2).mean()
            
        
        # normal loss
        if self.normal > 0:
            loss_normal += normal_loss(input_ny, df_y)
            
        metrics = {}
        if self.eikonal > 0:
            loss_es = loss_es / self.T
            loss += self.eikonal * ratio_i * loss_es
            metrics.update({'eikonal': loss_es})

        if self.nonmanifold > 0:
            loss_en = loss_en / self.T
            loss += self.nonmanifold * ratio_i * loss_en
            metrics.update({'nonmanifold': loss_en})

        if self.divergence > 0:
            loss_div = loss_div / self.T
            loss += self.divergence * ratio_v* loss_div
            metrics.update({'divergence': loss_div})

        if self.laplacian > 0:
            loss_lap = loss_lap / self.T
            loss += self.laplacian *ratio_v* loss_lap
            metrics.update({'laplacian': loss_lap})
        
        if self.level_set > 0:
            loss_lse = loss_lse / self.T
            loss += self.level_set * ratio_i * loss_lse
            metrics.update({'level_set': loss_lse})
        
        if self.manifold > 0:
            loss += self.manifold * ratio_i * loss_surf  / 2
            metrics.update({'manifold': loss_surf})
        
        if self.normal > 0:
            loss += self.normal * ratio_i * loss_normal / 3
            metrics.update({'normal': loss_normal / 3})
        
        if self.match > 0:
            loss += self.match * ratio_v * loss_match
            metrics.update({'match': loss_match})
        

        if self.surf_area > 0:
            loss += self.surf_area * ratio_i * loss_area / self.T
            metrics.update({'surf_area': loss_area / self.T})
        
        if self.bending > 0:
            loss_bending = loss_bending / self.T
            loss += self.bending * ratio_i * loss_bending
            metrics.update({'bending': loss_bending})
            
        if self.stretching > 0:
            loss_stretch = loss_stretch / self.T
            loss += self.stretching * ratio_i * loss_stretch
            metrics.update({'stretching': loss_stretch})
            
        if self.distortion > 0:
            loss_distort = loss_distort / self.T
            loss +=  self.distortion * ratio_v * loss_distort
            metrics.update({'distortion': loss_distort})


        metrics.update({'loss':loss})
        return loss, metrics
    
    
    '''
    for accelerate training, when ratio_i is zero, do not compute implicit net loss

    '''
    def get_velocity_loss(self, f_fn, params, fv_fn, params_v, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, index_pair, ratio):

        batch_size = input_x.shape[0] // 2
        
        points = input_x

        loss_lap, loss_div, loss_distort, loss_match, loss = 0., 0., 0.,0.,0.0 #smooth loss, divergence loss, distortion loss, matching loss, total loss
       
        for time in range(self.T):

            t = jnp.tile(time/self.T, (points.shape[0],1))

            v = partial(fv_fn, params_v)(points, t, index_pair)
            
            dv = vmap(self.get_v_jacobian, in_axes=(None, None, 0, 0, None))(fv_fn, params_v, points, t, index_pair)

            if (self.laplacian > 0.0):
                Hv = vmap(self.get_v_hessian, in_axes=(None, None, 0, 0, None))(fv_fn, params_v, points, t, index_pair)
                laplacian = jnp.trace(Hv, axis1=2, axis2=3)
                # loss_lap += jnp.linalg.norm(v - self.alpha * laplacian, axis=-1).mean()
                loss_lap += sq_norm(v - self.alpha * laplacian, axis=-1).mean()                   

            if (self.divergence > 0.0):
                div_v = jnp.trace(dv, axis1=1, axis2=2)
                loss_div += sdf_loss(div_v,'squared_l2')
            
            if self.distortion > 0:
                deformation = deformation_rate(dv)
                loss_distort += sdf_loss(deformation, 'squared_l2')
            
             # move points
            points = points + v
        
        #matching
        loss_match = match_loss(points[:batch_size,:], input_y[:batch_size,:])

        metrics = {}
        if self.divergence > 0:
            loss_div = loss_div / self.T
            loss += self.divergence * loss_div
            metrics.update({'divergence': loss_div})

        if self.laplacian > 0:
            loss_lap = loss_lap / self.T
            loss += self.laplacian * loss_lap
            metrics.update({'laplacian': loss_lap})
            
        if self.distortion > 0:
            loss_distort = loss_distort / self.T
            loss +=  self.distortion * loss_distort
            metrics.update({'distortion': loss_distort})
        
        if self.match > 0:
            loss += self.match * loss_match
            metrics.update({'match': loss_match})
        
        metrics.update({'loss':loss})
        return loss, metrics
    
    '''
    for accelerate training, when ratio_v is zero, do not compute velocity net loss

    '''
    def get_sdf_loss(self, f_fn, params, fv_fn, params_v, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, index_pair, ratio):
        
        batch_size = input_x.shape[0] // 2
        
        points = input_x

        loss, loss_lse, loss_es, loss_en, loss_n, loss_surf = 0., 0., 0., 0., 0., 0.
        loss_area = 0.0

        # fit the first
        t = jnp.tile(0., (input_x.shape[0],1))

        for time in range(self.T+1):

            t = jnp.tile(time/self.T, (points.shape[0],1))


            sdf, df, dt = vmap(self.get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, points, t)
            
            v = partial(fv_fn, params_v)(points)
            dv = vmap(self.get_v_jacobian, in_axes=(None, None, 0, 0, None))(fv_fn, params_v, points, t, index_pair)

            if (self.eikonal > 0) and (self.reinitial == 0):
                loss_es += ((1-soft_norm(df, axis=-1))**2).mean()
            
            if (self.nonmanifold > 0) and (self.reinitial == 0):
                df_s = vmap(self.get_surface_gradient, in_axes=(None, None, 0, 0))(f_fn, params, sample_local_x, t)
                loss_en += ((1-soft_norm(df_s, axis=-1))**2).mean()
                v_sample = partial(fv_fn, params_v)(sample_local_x)
                sample_local_x = sample_local_x + v_sample
                
            if self.surf_area > 0:
                loss_area += self.get_implicit_surface_area_loss(partial(f_fn, params), sample_global, t)
                
            if self.level_set > 0:
                if self.reinitial > 0:
                    R = (self.eikonal/self.level_set) * R_term(df, dv)
                    loss_lse += sdf_loss(dt + jnp.sum(df * v, axis=-1) + sdf*R,'squared_l2')
                else:
                    loss_lse += sdf_loss(dt + jnp.sum(df * v, axis=-1), 'squared_l2')

            if (self.manifold > 0):
                loss_surf += sdf_loss(sdf, distance_metric='l2')
            
            
            points = points + v
        
        # fit the second
        sdf, df, dt = vmap(self.get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, input_y, t)


        metrics = {}
        if self.eikonal > 0:
            loss_es = loss_es / self.T
            loss += self.eikonal * loss_es
            metrics.update({'eikonal': loss_es})

        if self.nonmanifold > 0:
            loss_en = loss_en / self.T
            loss += self.nonmanifold * loss_en
            metrics.update({'nonmanifold': loss_en})

        if self.level_set > 0:
            loss_lse = loss_lse / self.T
            loss += self.level_set * loss_lse
            metrics.update({'level_set': loss_lse})
        
        
        if self.normal > 0:
            loss_n += normal_loss(df, input_ny)
            loss += self.normal * loss_n / 3
            metrics.update({'normal': loss_n / 3})
        
        if self.manifold > 0:
            loss += self.manifold * loss_surf / (self.T-1)
            metrics.update({'onsurf': loss_surf/ (self.T-1)})
        
        if self.surf_area > 0:
            loss += self.surf_area * ratio * loss_area / self.T
            metrics.update({'surf_area': loss_area / self.T})

        metrics.update({'loss':loss})
        return loss, metrics

    
    
    
    def get_fine_tune_loss(self, f_fn, params, fv_fn, params_v, input_x, input_nx,  input_y, input_ny, sample_local_x, sample_local_y, sample_global, index, ratio):
        
        batch_size = input_x.shape[0] // 2
        
        features = self.get_feature(params_v=params_v, index_pair=index)
        
        loss = 0.0 #total loss
        loss_bending, loss_stretch, loss_surf, loss_normal, loss_area = 0.0, 0.0, 0.0, 0.0, 0.0 #normal deformation loss, strectch loss, on surface loss, normal loss, minimum area loss
        
        loss_en, loss_es = 0.0, 0.0 #eikonal on non-surf, eikonal on surf
        
        points = input_x
        
        sample_local_size = sample_local_x.shape[0]
        condition = jnp.tile(features, (points.shape[0], 1))
        
        df = input_nx
        for time in range(self.T+1):
            
            # time step
            t = jnp.tile(time/self.T,  (points.shape[0],1))
            
            v = partial(fv_fn, params_v)(points, t, index)
            
            dv = vmap(self.get_v_jacobian, in_axes=(None, None, 0, 0, None))(fv_fn, params_v, points, t, index)
            
            #save last step normal
            df_last = df

            # implicit
            sdf, df, dt = vmap(self.get_full_gradient, in_axes=(None, None, 0, 0, 0))(f_fn, params, points, t, condition)
            
            
            if (self.manifold > 0):
                loss_surf += sdf_loss(sdf, distance_metric='l2')
            
            
            if self.nonmanifold > 0:
                sdf_s, df_s, dt_s = vmap(self.get_full_gradient, in_axes=(None, None, 0, 0, 0))(f_fn, params, sample_local_x, t, condition)
                v_sample = partial(fv_fn, params_v)(sample_local_x, t, index)
                loss_en += ((1-soft_norm(df_s, axis=-1))**2).mean()
                sample_local_x = sample_local_x + v_sample
            
            # area loss
            if self.surf_area > 0:
                loss_area += self.get_implicit_surface_area_loss(partial(f_fn, params), sample_global, jnp.tile(time,(sample_global.shape[0],1)), condition=jnp.tile(features, (sample_global.shape[0],1)))
            
            # ekional loss
            if self.eikonal > 0:
                loss_en += ((1-soft_norm(df, axis=-1))**2).mean()
            
            
            # normal loss
            if (self.normal > 0) and (time == 0):
                loss_normal += normal_loss(input_nx, df)
            
            if (self.bending > 0) and (time > 0):
                loss_bending += deformation_gradient(df_last, df, dv)

            if self.stretching > 0:
                loss_stretch += stretching_loss(df, dv)

            
            points = points + v
            
    
        # fit second
        t = jnp.tile(1.0, (input_y.shape[0],1))
        
        sdf_y, df_y, dt = vmap(self.get_full_gradient, in_axes=(None, None, 0, 0, 0))(f_fn, params, input_y, t, condition)
        
        if self.manifold > 0:
            loss_surf += sdf_loss(sdf_y, distance_metric='l2')
        
        if self.nonmanifold > 0:
            sdf_s, df_s, dt_s = vmap(self.get_full_gradient, in_axes=(None, None, 0, 0, 0))(f_fn, params, sample_local_y, t, condition)
            loss_en += ((1-soft_norm(df_s, axis=-1))**2).mean()
            
        
        # normal loss
        if self.normal > 0:
            loss_normal += normal_loss(input_ny, df_y)
            
        if self.eikonal > 0:
            loss_es += ((1-soft_norm(df_y, axis=-1))**2).mean()
        
        metrics = {}
        
        if self.manifold > 0:
            loss += self.manifold * 5 * loss_surf / self.T
            metrics.update({'manifold': loss_surf})
        
        if self.nonmanifold > 0:
            loss += self.nonmanifold * loss_en / self.T
            metrics.update({'nonmanifold': loss_en})
            
        if self.normal > 0:
            loss += self.normal * loss_normal / 2
            metrics.update({'normal': loss_normal})
        
        if self.surf_area > 0:
            loss += self.surf_area *  loss_area / self.T
            metrics.update({'surf_area': loss_area})
        
        if self.eikonal > 0:
            loss += self.eikonal *  loss_es / self.T
            metrics.update({'eikonal': loss_es})

        if self.stretching > 0:
            loss_stretch = loss_stretch / self.T
            loss += self.stretching *  loss_stretch
            metrics.update({'stretching': loss_stretch})
            
        if self.bending > 0:
            loss_bending = loss_bending / self.T
            loss += self.bending * loss_bending
            metrics.update({'bending': loss_bending})

    
        return loss, metrics
        

    
    @partial(jit, static_argnames=("self"))
    def train_step(self, input_x, input_nx,  input_y, input_ny, sample_local_x, sample_local_y, sample_global, index, ratio, implicit_state: TrainState, velocity_state: TrainState):
        
        (loss, metrics), (grads_f, grads_v) = jax.value_and_grad(self.get_loss, argnums=(1,3), has_aux=True)(implicit_state.apply_fn, implicit_state.params, velocity_state.apply_fn, velocity_state.params, input_x, input_nx,  input_y, input_ny, sample_local_x, sample_local_y, sample_global, index, ratio)
        
        implicit_state, nan_grads = safe_apply_grads(implicit_state, grads_f)
        
        velocity_state, nan_grads = safe_apply_grads(velocity_state, grads_v)

        return loss, metrics, implicit_state, velocity_state
    
    
    
    @partial(jit, static_argnames=("self"))
    def train_step_frozen_sdf(self, input_x, input_nx,  input_y, input_ny, sample_local_x, sample_local_y, sample_global, index, ratio, implicit_state: TrainState, velocity_state: TrainState):
        
        (loss, metrics), grads_v = jax.value_and_grad(self.get_velocity_loss, argnums=3, has_aux=True)(implicit_state.apply_fn, implicit_state.params, velocity_state.apply_fn, velocity_state.params, input_x, input_nx,  input_y, input_ny, sample_local_x, sample_local_y, sample_global, index, ratio)
        
        velocity_state, nan_grads = safe_apply_grads(velocity_state, grads_v)

        return loss, metrics, implicit_state, velocity_state
    

    
    @partial(jit, static_argnames=("self"))
    def train_step_fine_tune_sdf(self, input_x, input_nx,  input_y, input_ny, sample_local_x, sample_local_y, sample_global, index, ratio, implicit_state: TrainState, velocity_state: TrainState):
        
        (loss, metrics), grads_f = jax.value_and_grad(self.get_fine_tune_loss, argnums=1, has_aux=True)(implicit_state.apply_fn, implicit_state.params, velocity_state.apply_fn, velocity_state.params, input_x, input_nx,  input_y, input_ny, sample_local_x, sample_local_y, sample_global, index, ratio)
        
        implicit_state, nan_grads = safe_apply_grads(implicit_state, grads_f)
        
        return loss, metrics, implicit_state, velocity_state




