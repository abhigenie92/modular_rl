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
import gym
import ipdb,pdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument("--plot",action="store_true")
    parser.add_argument("--exp_name",required=True)
    # experiment parse
    exp_parser = argparse.ArgumentParser()
    exp_parser.add_argument('--exp_name')
    exp_name_arg=exp_parser.parse_args([sys.argv[-1]])

    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help','--exp_name')])
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
    agent = agent_ctor(env.observation_space, env.action_space, cfg)
    if args.use_hdf:
        hdf, diagnostics = prepare_h5_file(args)
    gym.logger.setLevel(logging.WARN)
    pol_kl_after=[]
    pol_kl_before=[]
    pol_ent_before=[]
    pol_ent_after=[]
    EpRewMean=[]
    var1=[]
    var2=[]
    COUNTER = 0

    exp_fldr='.'+os.sep+'exp_results'+os.sep
    if not os.path.exists(exp_fldr+exp_name_arg.exp_name):
        os.makedirs(exp_fldr+exp_name_arg.exp_name)
    fldr_path=exp_fldr+exp_name_arg.exp_name+os.sep

    def callback(stats,agent,variance,debug=False):
        # save stats
        pol_kl_after.append(stats['pol_kl_after'])
        pol_kl_before.append(stats['pol_kl_before'])
        pol_ent_before.append(stats['pol_ent_before'])
        pol_ent_after.append(stats['pol_ent_after'])
        EpRewMean.append(stats['EpRewMean'])
        var1.append(variance[0])
        var2.append(variance[1])
        wts_biases=agent.baseline.reg.net.get_weights()

        
        wts=wts_biases[::2]
        biases=wts_biases[1::2]
        
        global COUNTER
        COUNTER += 1
        print "*********** Iteration %i ****************" % COUNTER

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
        plt.savefig(fldr_path+name+'.png')
        plt.close() 
    def plot_var_save(var1,var2,ylabel="Variance across layer",name='var'):
        num_iters=range(len(var1))
        plt.plot(num_iters,var1,"Layer 1")
        plt.plot(num_iters,var2,"Layer 2")
        plt.legend(shadow=True)
        plt.xlabel('Iteration')
        plt.ylabel(ylabel)
        plt.savefig(fldr_path+name+'.png')
        plt.close() 

    '''
    agent.baseline.reg.net
    '''
    print matplotlib.get_backend()
    plot_save(pol_kl_after,'KL loss after policy update','pol_kl_after')
    plot_save(pol_kl_before,'KL loss before policy update','pol_kl_before')
    plot_save(pol_ent_before,'Policy entropy before policy update','pol_kl_before')
    plot_save(pol_ent_after,'Policy entropy after policy update','pol_kl_before')
    plot_save(EpRewMean,'Episode Reward Mean','EpRewMean')
    #plot_var_save(var1,var2)
    
    

    if args.use_hdf:
        hdf['env_id'] = env_spec.id
        try: hdf['env'] = np.array(cPickle.dumps(env, -1))
        except Exception: print "failed to pickle env" #pylint: disable=W0703
    env.close()
