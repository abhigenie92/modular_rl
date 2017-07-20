#!/usr/bin/env python
"""
This script runs a policy gradient algorithm
"""

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib import pyplot as plt

from gym.envs import make
from modular_rl import *
import argparse, sys, cPickle
from tabulate import tabulate
import shutil, os, logging
import gym,sys
#import ipdb,pdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument("--plot",action="store_true")
    parser.add_argument("--exp_name",required=True)
    parser.add_argument("--wt_scale",required=True)
    # experiment,wt_scale parse
    parser_hyp = argparse.ArgumentParser()
    parser_hyp.add_argument('--exp_name')
    parser_hyp.add_argument('--wt_scale')
    parser_hyp_args=parser_hyp.parse_args(sys.argv[-2:])
    exp_name=parser_hyp_args.exp_name
    wt_scale=int(parser_hyp_args.wt_scale)

    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help','--exp_name','--wt_scale')])
    env = make(args.env)
    env_spec = env.spec
    mondir = args.outfile + ".dir"
    if os.path.exists(mondir): shutil.rmtree(mondir)
    os.mkdir(mondir)
    env = gym.wrappers.Monitor(env, mondir, video_callable=None if args.video else VIDEO_NEVER)
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    if args.timestep_limit == 0:
        args.timestep_limit = env_spec.timestep_limit
    cfg = args.__dict__
    np.random.seed(args.seed)
    agent = agent_ctor(env.observation_space, env.action_space, cfg, wt_scale)
    if args.use_hdf:
        hdf, diagnostics = prepare_h5_file(args)
    gym.logger.setLevel(logging.WARN)
    pol_kl_after=[]
    pol_kl_before=[]
    pol_ent_before=[]
    pol_ent_after=[]
    EpRewMean=[]
    grad_w_norms_layer1_mean=[]
    grad_w_norms_layer2_mean=[]
    grad_w_norms_layer1_var=[]
    grad_w_norms_layer2_var=[]
    grad_w_diff_1=[]
    grad_w_diff_2=[]
    var1=[]
    var2=[]
    COUNTER = 0

    exp_fldr='.'+os.sep+'exp_results'+os.sep
    if not os.path.exists(exp_fldr+exp_name):
        os.makedirs(exp_fldr+exp_name)
    fldr_path=exp_fldr+exp_name+os.sep
    grad_prev_1=None
    grad_prev_2=None

    def callback(stats,agent,variance,grad_w_norms,debug=False):
        # save stats
        pol_kl_after.append(stats['pol_kl_after'])
        pol_kl_before.append(stats['pol_kl_before'])
        pol_ent_before.append(stats['pol_ent_before'])
        pol_ent_after.append(stats['pol_ent_after'])
        EpRewMean.append(stats['EpRewMean'])
        var1.append(variance[0])
        var2.append(variance[1])
        wts_biases=agent.baseline.reg.net.get_weights()
        

        grad_w_norms_layer1_mean.append(np.mean(grad_w_norms[0]))
        grad_w_norms_layer2_mean.append(np.mean(grad_w_norms[1]))
        grad_w_norms_layer1_var.append(np.var(grad_w_norms[0]))
        grad_w_norms_layer2_var.append(np.var(grad_w_norms[1]))
        global COUNTER,grad_prev_1,grad_prev_2
        if COUNTER>0:
            diff=(grad_w_norms[0]-grad_prev_1)/grad_prev_1
            diff_mean=np.mean(np.absolute(diff))
            grad_w_diff_1.append(diff_mean)

            diff=(grad_w_norms[1]-grad_prev_2)/grad_prev_2
            diff_mean=np.mean(np.absolute(diff))
            grad_w_diff_2.append(diff_mean)
        grad_prev_1=grad_w_norms[0]
        grad_prev_2=grad_w_norms[1]

        
        wts=wts_biases[::2]
        biases=wts_biases[1::2]
        
        COUNTER += 1
        print COUNTER,
        sys.stdout.flush()

        #print "*********** Iteration %i ****************" % COUNTER

        # Print stats
        if debug:
            print tabulate(filter(lambda (k,v) : np.asarray(v).size==1, stats.items())) #pylint: disable=W0110
            # Store to hdf5
            
            if args.use_hdf:
                for (stat,val) in stats.items():
                    if np.asarray(val).ndim==0:
                        diagnostics[stat].append(val)
                    else:
                        assert val.ndim == 1
                        diagnostics[stat].extend(val)
                if args.snapshot_every and ((COUNTER % args.snapshot_every==0) or (COUNTER==args.n_iter)):
                    hdf['/agent_snapshots/%0.4i'%COUNTER] = np.array(cPickle.dumps(agent,-1))
            # Plot
            if args.plot:
                animate_rollout(env, agent, min(500, args.timestep_limit))

    run_policy_gradient_algorithm(env, agent, callback=callback, usercfg = cfg,debug=False)
    def plot_save(data,ylabel,name):
        data=[x.tolist() for x in data] # scalar to values in list
        plt.plot(data)
        plt.xlabel('Iteration')
        plt.ylabel(ylabel)
        plt.show()
        plt.savefig(fldr_path+name+'.png')
        plt.close() 
    def plot_var_save(var1,var2,ylabel="Variance across activations in a layer",name='var'):
        num_iters=range(len(var1))
        plt.plot(num_iters,var1,label="Layer 1")
        plt.plot(num_iters,var2,label="Layer 2")
        plt.legend(shadow=True)
        plt.xlabel('Iteration')
        plt.ylabel(ylabel)
        plt.show()
        plt.savefig(fldr_path+name+'.png')
        plt.close() 

    '''
    agent.baseline.reg.net
    '''
    plot_save(EpRewMean,'Episode Reward Mean','EpRewMean')
    print "KL Loss"
    plot_save(pol_kl_after,'KL loss after policy update','pol_kl_after')
    plot_save(pol_kl_before,'KL loss before policy update','pol_kl_before')
    print "Policy Entropy"
    plot_save(pol_ent_before,'Policy entropy before policy update','pol_kl_before')
    plot_save(pol_ent_after,'Policy entropy after policy update','pol_kl_before')
    print "Gradient Norms statistics"
    plot_save(grad_w_norms_layer1_mean,'Gradient_w Norm-2 Layer1 Mean','Gradw1_mean')
    plot_save(grad_w_norms_layer2_mean,'Gradient_w Norm-2 Layer2 Mean','Gradw2_mean')
    plot_save(grad_w_norms_layer1_var,'Gradient_w Norm-2 Variance','Gradw1_var')
    plot_save(grad_w_norms_layer1_var,'Gradient_w Norm-2 Variance','Gradw2_var')
    print "Gradient Norm Differences between timesteps"
    plot_save(grad_w_diff_1,'Gradient W Layer1 Diff','Gradw1_diff')
    plot_save(grad_w_diff_2,'Gradient W Layer2 Diff','Gradw2_diff')
    print "Variance in activations"
    plot_var_save(var1,var2)
    
    

    if args.use_hdf:
        hdf['env_id'] = env_spec.id
        try: hdf['env'] = np.array(cPickle.dumps(env, -1))
        except Exception: print "failed to pickle env" #pylint: disable=W0703
    env.close()
