"""Deterministic matched-observation action metrics, paired cluster bootstrap, plots."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
R=Path(__file__).resolve().parent;O=R/'results';O.mkdir(exist_ok=True);F=O/'figures';F.mkdir(exist_ok=True)
plt.style.use(str(R/'academic.mplstyle'))
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'figure.dpi':140,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False,'grid.alpha':.2,'grid.linewidth':.5})
models={'pi05_baseline7500':('Baseline 7.5k',R/'raw/A800/run/matched_20261012/baseline7500'),'c01_lora97500':('C01 LoRA 97.5k',R/'raw/A800/run/matched_20261012/lora97500'),'c01_fullft60000':('C01 full FT 60k',R/'raw/A100/run/matched_20261012/fullft60000')}
colors={'pi05_baseline7500':'#929292','c01_lora97500':'#7e9aaa','c01_fullft60000':'#a07d70'}
styles={'pi05_baseline7500':'--','c01_lora97500':'-.','c01_fullft60000':'-'}
names=['L shoulder pitch','L shoulder roll','L shoulder yaw','L elbow','L wrist roll','L wrist pitch','L wrist yaw','R shoulder pitch','R shoulder roll','R shoulder yaw','R elbow','R wrist roll','R wrist pitch','R wrist yaw','L gripper','R gripper']
task_names=['Move black cup','Place chrysanthemum','Grasp white cup','Steady black cup','Pour water / home','Place red dates','Place lemon slices']
arrays={k:{n:z[n] for n in z.files} for k,(_,p) in models.items() for z in [np.load(p/'predictions.npz')]}
ref=arrays['pi05_baseline7500'];
for k in ['c01_lora97500','c01_fullft60000']:assert np.array_equal(ref['canonical_observation_sha256'],arrays[k]['input_sha256'])
assert np.array_equal(arrays['c01_lora97500']['input_sha256'],arrays['c01_fullft60000']['input_sha256'])
gt=ref['targets'];mask=ref['action_valid'];task=ref['task_indices'];sources=ref['source_episode_indices'];state=ref['raw_states'];unique_sources=np.unique(sources)
for key,d in arrays.items():
 assert d['predictions'].shape==(200,30,16) and np.isfinite(d['predictions']).all()
 for name in ['indices','task_indices','action_valid','source_episode_indices','subtask_episode_indices','frame_indices','prompts']:assert np.array_equal(ref[name],d[name]),(key,name)
 for name in ['targets','raw_states','future_states']:assert np.allclose(ref[name],d[name],atol=1e-6,rtol=0),(key,name)
 raw_gt=gt.copy();gt=np.where(mask[...,None],gt,np.nan)
stats=json.loads((R/'raw/A100/assets/norm_stats.json').read_text());stats=stats.get('norm_stats',stats);action_stats=stats['actions'];scale=np.asarray(action_stats['q99'])-np.asarray(action_stats['q01'])+1e-6
from matplotlib.backends.backend_pdf import PdfPages
pdf_all=PdfPages(O/'ALL_FIGURES.pdf')
catalog=[]
def save(fig,name,claim):
 pdf_all.savefig(fig,bbox_inches='tight');fig.savefig(F/f'{name}.png',bbox_inches='tight');fig.savefig(F/f'{name}.pdf',bbox_inches='tight');plt.close(fig);catalog.append({'name':name,'surface_class':'internal_review','claim':claim,'png':str(F/f'{name}.png'),'pdf':str(F/f'{name}.pdf'),'generator':str(Path(__file__)),'source_data':[str(p/'predictions.npz') for _,p in models.values()],'visual_review':'pending'})
def dumpcsv(name,rows):
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def core(p,y):
 e=p-y;a=np.abs(e);corr=[];r2=[]
 for j in range(16):
  x=p[...,j].ravel();z=y[...,j].ravel();sel=np.isfinite(x)&np.isfinite(z);x=x[sel].astype(np.float64);z=z[sel].astype(np.float64);corr.append(float(np.corrcoef(x,z)[0,1]) if np.std(x)>1e-12 and np.std(z)>1e-12 else None);den=np.sum((z-z.mean())**2);r2.append(float(1-np.sum((x-z)**2)/den) if den>1e-12 else None)
 return {'mae':float(np.nanmean(a)),'rmse':float(np.sqrt(np.nanmean(e*e))),'mse':float(np.nanmean(e*e)),'signed_bias':float(np.nanmean(e)),'median_abs_error':float(np.nanmedian(a)),'p90_abs_error':float(np.nanquantile(a,.9)),'p95_abs_error':float(np.nanquantile(a,.95)),'p99_abs_error':float(np.nanquantile(a,.99)),'max_abs_error':float(np.nanmax(a)),'range_normalized_mae':float(np.nanmean(a/scale)),'first_step_mae':float(np.nanmean(a[:,0])),'last_step_mae':float(np.nanmean(a[:,-1])),'per_dim_mae':np.nanmean(a,axis=(0,1)).tolist(),'per_dim_rmse':np.sqrt(np.nanmean(e*e,axis=(0,1))).tolist(),'per_dim_bias':np.nanmean(e,axis=(0,1)).tolist(),'per_dim_r2':r2,'per_dim_pearson':corr,'horizon_mae':np.nanmean(a,axis=(0,2)).tolist(),'horizon_arm_mae':np.nanmean(a[:,:,:14],axis=(0,2)).tolist(),'horizon_gripper_mae':np.nanmean(a[:,:,14:],axis=(0,2)).tolist(),'horizon_rmse':np.sqrt(np.nanmean(e*e,axis=(0,2))).tolist(),'per_observation_mae':np.nanmean(a,axis=(1,2)).tolist()}
groups={'arms':slice(0,14),'left_arm':slice(0,7),'right_arm':slice(7,14),'grippers':slice(14,16),'left_gripper':slice(14,15),'right_gripper':slice(15,16)}
result={'protocol':{'observations':200,'unique_dataset_indices':len(np.unique(ref['indices'])),'unique_canonical_observations':len(np.unique(ref['canonical_observation_sha256'])),'source_episodes':len(unique_sources),'subtask_episodes':len(np.unique(ref['subtask_episode_indices'])),'horizon':30,'fps':30,'action_dim':16,'anchor_sha256':hashlib.sha256(ref['indices'].tobytes()).hexdigest(),'raw_rgb_pixels_state_tokens_match_all_models':True,'c01_input_tensors_bitwise_equal':True,'baseline_normalized_image_roundoff_atol':2e-7,'baseline_input_audit':'Original uint8 pixels and state/tokens match; baseline JAX image normalization differs from C01 NumPy by <=one float32 ULP; original baseline tensor hashes preserved','targets_match_atol':1e-6,'mask_all_valid':bool(mask.all()),'valid_action_positions':int(mask.sum()),'padded_action_positions_excluded':int((~mask).sum()),'temporal_mask':'only differences with all involved timesteps valid','checkpoint_selection':'LoRA selected97500 after100k complete; FT interim60000 while100k continues','bootstrap':'2000 paired cluster resamples of original source episodes; fixed seed20261012','units':'native motor position units; arm angles in rad; gripper separately reported','limitations':['Selected heldout reused for checkpoint selection; not independent test.','BaselineH32 uses first30; C01H30; latent noise dimensions differ.','Unequal training samples/batches/LR budgets.','Temporal metrics describe target chunks, not executed physical vibration.','Full FT 100k task has not completed; evaluated60000 checkpoint only.']},'models':{},'paired_bootstrap':{}}
rows=[];hrows=[];jrows=[];obsrows=[];trows=[];grows=[]
for key,(label,path) in models.items():
 p=arrays[key]['predictions'];e=p-gt;a=np.abs(e);m=core(p,gt);m['groups']={n:{'mae':float(np.nanmean(a[...,s])),'rmse':float(np.sqrt(np.nanmean(e[...,s]**2))),'range_normalized_mae':float(np.nanmean(a[...,s]/scale[s]))} for n,s in groups.items()}
 m['per_task']={str(t):core(p[task==t],gt[task==t]) for t in np.unique(task)}
 m['temporal']={}
 for group,s in groups.items():
  x=np.where(mask[...,None],p,np.nan)[...,s];y=gt[...,s];v=np.diff(x,axis=1);vgt=np.diff(y,axis=1);acc=np.diff(v,axis=1);agt=np.diff(vgt,axis=1);jerk=np.diff(acc,axis=1)
  # Ignore velocities smaller than0.001 native units when counting reversals.
  significant=(np.abs(v[:,1:])>.001)&(np.abs(v[:,:-1])>.001);rev=(v[:,1:]*v[:,:-1]<0)&significant
  m['temporal'][group]={'delta_mae':float(np.nanmean(np.abs(v-vgt))),'delta_rmse':float(np.sqrt(np.nanmean(((v-vgt)**2)))),'second_difference_mae':float(np.nanmean(np.abs(acc-agt))),'mean_abs_delta':float(np.nanmean(np.abs(v))),'gt_mean_abs_delta':float(np.nanmean(np.abs(vgt))),'mean_abs_second_difference':float(np.nanmean(np.abs(acc))),'gt_mean_abs_second_difference':float(np.nanmean(np.abs(agt))),'mean_abs_third_difference':float(np.nanmean(np.abs(jerk))),'mean_reversals_per_joint_chunk':float(rev.sum()/len(p)/x.shape[-1]),'first_action_state_jump_mae':float(np.nanmean(np.abs(x[:,0]-state[:,s]))),'gt_first_action_state_jump_mae':float(np.nanmean(np.abs(y[:,0]-state[:,s])))}
 m['future_measured_state_mae']=float(np.nanmean(np.where(mask[...,None],np.abs(p-ref['future_states']),np.nan)));m['prefix_windows']={str(h):{'mae':float(np.nanmean(np.abs(p[:,:h]-gt[:,:h]))),'rmse':float(np.sqrt(np.nanmean((p[:,:h]-gt[:,:h])**2))),'arm_mae':float(np.nanmean(np.abs(p[:,:h,:14]-gt[:,:h,:14]))),'gripper_mae':float(np.nanmean(np.abs(p[:,:h,14:]-gt[:,:h,14:])))} for h in [1,5,8,10,16,20,30]};m['note_future_state']='Diagnostic only: predicted commanded positions versus measured states at corresponding future timestamps; not the supervised action metric.'
 disp=p[:,-1,:14]-state[:,:14];truthdisp=gt[:,-1,:14]-state[:,:14];active=np.abs(truthdisp)>.02;m['terminal_arm_motion_sign_agreement_threshold_0p02_rad']=float((np.sign(disp[active])==np.sign(truthdisp[active])).mean());m['terminal_arm_active_components']=int(active.sum())
 result['models'][key]=m
 rows.append({'model':label,**{n:m[n] for n in ['mae','rmse','mse','range_normalized_mae','median_abs_error','p95_abs_error','p99_abs_error','max_abs_error','first_step_mae','last_step_mae']},'arm_mae':m['groups']['arms']['mae'],'gripper_mae':m['groups']['grippers']['mae'],'delta_arm_mae':m['temporal']['arms']['delta_mae']})
 for g,v in m['groups'].items():grows.append({'model':label,'group':g,**v,**m['temporal'][g]})
 for t,v in m['per_task'].items():trows.append({'model':label,'task_index':int(t),'task':task_names[int(t)],'observations':int((task==int(t)).sum()),'mae':v['mae'],'rmse':v['rmse'],'range_normalized_mae':v['range_normalized_mae']})
 for j,name in enumerate(names):jrows.append({'model':label,'joint_index':j,'joint':name,'mae':m['per_dim_mae'][j],'rmse':m['per_dim_rmse'][j],'bias':m['per_dim_bias'][j],'r2':m['per_dim_r2'][j],'pearson':m['per_dim_pearson'][j]})
 for step in range(30):hrows.append({'model':label,'step':step,'time_s':step/30,'mae':m['horizon_mae'][step],'rmse':m['horizon_rmse'][step],'arm_mae':m['horizon_arm_mae'][step],'gripper_mae':m['horizon_gripper_mae'][step]})
 for n,v in enumerate(m['per_observation_mae']):obsrows.append({'model':label,'ordinal':n,'dataset_index':int(ref['indices'][n]),'source_episode':int(sources[n]),'subtask_episode':int(ref['subtask_episode_indices'][n]),'frame':int(ref['frame_indices'][n]),'task':int(task[n]),'mae':v})
# Paired bootstrap weighted by valid elements after resampling original episodes.
rng=np.random.default_rng(20261012);samples=rng.integers(0,len(unique_sources),(2000,len(unique_sources)));cirows=[]
for key in models:
 ab=np.abs(arrays[key]['predictions']-gt);ss=(arrays[key]['predictions']-gt)**2
 counts=np.array([np.isfinite(ab[sources==s]).sum() for s in unique_sources]);sums=np.array([np.nansum(ab[sources==s]) for s in unique_sources]);squares=np.array([np.nansum(ss[sources==s]) for s in unique_sources]);weights=counts[samples].sum(1);vals=sums[samples].sum(1)/weights;rm=np.sqrt(squares[samples].sum(1)/weights)
 result['models'][key]['cluster_bootstrap_ci95']={'mae':np.quantile(vals,[.025,.975]).tolist(),'rmse':np.quantile(rm,[.025,.975]).tolist()}
 result['models'][key]['bootstrap_mae_values']=vals.tolist();result['models'][key]['bootstrap_rmse_values']=rm.tolist()
for target in ['c01_lora97500','c01_fullft60000']:
 for baseline in ['pi05_baseline7500','c01_lora97500']:
  if target==baseline:continue
  u=np.array(result['models'][target]['bootstrap_mae_values']);v=np.array(result['models'][baseline]['bootstrap_mae_values']);diff=u-v;ratio=(1-u/v)*100;tr=np.array(result['models'][target]['bootstrap_rmse_values']);br=np.array(result['models'][baseline]['bootstrap_rmse_values']);ci={'delta_rmse':float(result['models'][target]['rmse']-result['models'][baseline]['rmse']),'delta_rmse_ci95':np.quantile(tr-br,[.025,.975]).tolist(),'rmse_reduction_pct':float((1-result['models'][target]['rmse']/result['models'][baseline]['rmse'])*100),'rmse_reduction_pct_ci95':np.quantile((1-tr/br)*100,[.025,.975]).tolist(),'delta_mae':float(result['models'][target]['mae']-result['models'][baseline]['mae']),'delta_mae_ci95':np.quantile(diff,[.025,.975]).tolist(),'mae_reduction_pct':float((1-result['models'][target]['mae']/result['models'][baseline]['mae'])*100),'mae_reduction_pct_ci95':np.quantile(ratio,[.025,.975]).tolist()};result['paired_bootstrap'][target+'_vs_'+baseline]=ci;cirows.append({'model':models[target][0],'reference':models[baseline][0],'delta_mae':ci['delta_mae'],'delta_ci_low':ci['delta_mae_ci95'][0],'delta_ci_high':ci['delta_mae_ci95'][1],'mae_reduction_pct':ci['mae_reduction_pct'],'reduction_ci_low':ci['mae_reduction_pct_ci95'][0],'reduction_ci_high':ci['mae_reduction_pct_ci95'][1]})
for m in result['models'].values():m.pop('bootstrap_mae_values');m.pop('bootstrap_rmse_values')
hold=np.broadcast_to(state[:,None,:],gt.shape);result['hold_current_state_reference']=core(hold,gt);result['hold_current_state_reference']['note']='Analytic diagnostic, no neural inference; holds measured current motor positions over30 steps. Not a closed-loop task policy.'
(O/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
for name,data in [('overall.csv',rows),('by_task.csv',trows),('by_joint.csv',jrows),('by_horizon.csv',hrows),('by_observation.csv',obsrows),('by_group_temporal.csv',grows),('paired_bootstrap.csv',cirows)]:dumpcsv(name,data)
contract={'baseline':'pi0.5 official subtask LoRA7500','route':'attach historical verified predictions after200-row raw RGB/state/token equivalence audit (image floats<=2e-7)','source_sha256':json.loads((models['pi05_baseline7500'][1]/'INPUT_AUDIT.json').read_text())['baseline_source_npz_sha256'],'accepted':True,'trusted_for':'selected-holdout open-loop physical action comparison only','metric_definitions':{'mae':'mean absolute predicted-minus-demonstration action over valid200x30x16 elements','rmse':'square root of mean squared action error','range_normalized_mae':'mean abs error divided by train action(q99-q01+1e-6), per dimension','delta_mae':'mean abs difference of consecutive predicted action delta and demonstration delta','second_difference_mae':'mean abs difference of consecutive action second differences','bootstrap_ci95':'paired source-episode cluster bootstrap percentile95%,2000 replicates; no independent test claim'},'direction':'lower is better for target-error metrics; smoothness magnitude alone is not success','paired_observation_verified':True,'float_image_roundoff_note':'Original baseline JAX versus C01 NumPy normalized float images differ<=2e-7; exact reconstructed uint8 RGB and nonimage inputs. Both hashes preserved.','unit_note':result['protocol']['units']};contract['historical_input_archival_limitation']='Original baseline prediction package does not contain per-observation tensor hashes; audit hashes reconstruct current original preprocessing. Baseline neural inference was not rerun.';(O/'metric_contract.json').write_text(json.dumps(contract,indent=2))
# Overall accuracy and group comparisons.
fig,axs=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
for ax,field,title in zip(axs,['mae','rmse','range_normalized_mae'],['Action MAE','Action RMSE','Range-normalized MAE']):
 for n,(key,(label,_)) in enumerate(models.items()):
  v=result['models'][key][field];ax.bar(n,v,color=colors[key],width=.6)
  if field in ['mae','rmse']:
   low,high=result['models'][key]['cluster_bootstrap_ci95'][field];ax.errorbar(n,v,yerr=[[v-low],[high-v]],color='#444444',capsize=3,fmt='none',lw=1)
  ax.text(n,(high if field in ['mae','rmse'] else v)+ax.get_ylim()[1]*.025,f'{v:.4f}',ha='center',fontsize=8)
 ax.set_xticks(range(3),['Baseline\n7.5k','LoRA\n97.5k','Full FT\n60k']);ax.set_title(title);ax.set_ylabel('Native position error' if field!='range_normalized_mae' else 'Fraction of training range');ax.set_ylim(0,ax.get_ylim()[1]*1.18)
fig.suptitle('Matched current observations: 200 evaluation rows, 30-step / 16-joint actions\nError bars: 95% cluster bootstrap intervals (20 source episodes)',fontsize=10)
save(fig,'01_overall_accuracy','Overall teacher-observation physical action error, no closed-loop success inference')
fig,axs=plt.subplots(1,3,figsize=(11,3.1),layout='constrained')
for ax,field,title in zip(axs,['horizon_mae','horizon_arm_mae','horizon_gripper_mae'],['All16 joints','Arm14 joints','Two grippers']):
 for key,(label,_) in models.items():ax.plot(np.arange(30)/30,result['models'][key][field],label=label,color=colors[key],ls=styles[key],lw=1.7)
 ax.plot(np.arange(30)/30,result['hold_current_state_reference'][field],label='Hold current state',color='#b6aca6',ls=':',lw=1.2)
 ax.set_title(title);ax.set_xlabel('Prediction offset (s at30Hz)');ax.set_ylabel('MAE (native units)');ax.grid(axis='y',alpha=.18)
axs[0].legend(frameon=False);save(fig,'02_horizon_error','Error versus prediction offset, split arms/grippers')
fig,axs=plt.subplots(2,1,figsize=(10,5.6),layout='constrained')
for k,(label,_) in models.items():
 n=list(models).index(k);offset=(n-1)*.23
 axs[0].bar(np.arange(7)+offset,[result['models'][k]['per_task'][str(t)]['mae'] for t in range(7)],.22,label=label,color=colors[k]);axs[1].plot(np.arange(16),result['models'][k]['per_dim_mae'],marker=['o','s','^'][n],ms=3,lw=1.4,ls=styles[k],color=colors[k],label=label)
axs[0].set_xticks(range(7),['Move\nblack cup','Place\nchrysanthemum','Grasp\nwhite cup','Steady\nblack cup','Pour water\n/ home','Place\nred dates','Place\nlemon slices']);axs[0].set_ylabel('Action MAE');axs[0].legend(ncol=3,frameon=False);axs[1].set_xticks(range(16),names,rotation=50,ha='right',fontsize=8);axs[1].set_ylabel('Joint MAE');axs[1].axvline(13.5,color='#cccccc',lw=.8);save(fig,'03_task_and_joint_error','Per-subtask and per-motor error; grippers shown separately by position in joint plot')
fig,axs=plt.subplots(1,2,figsize=(8.4,3.3),layout='constrained')
for key,(label,_) in models.items():
 v=np.sort(result['models'][key]['per_observation_mae']);axs[0].plot(v,np.arange(1,201)/200,label=label,color=colors[key],ls=styles[key]);t=result['models'][key]['temporal']['arms'];n=list(models).index(key);axs[1].bar(n-.16,t['delta_mae'],.3,color=colors[key]);axs[1].bar(n+.16,t['second_difference_mae'],.3,color=colors[key],alpha=.55,hatch='//')
axs[0].set_xlabel('Per-observation action MAE');axs[0].set_ylabel('Fraction of200 observations');axs[0].legend(frameon=False);axs[1].set_xticks(range(3),['Baseline7.5k','LoRA97.5k','FT60k']);axs[1].set_ylabel('Arm action-difference MAE (rad)');axs[1].set_title('Solid: first difference; hatched: second difference');save(fig,'04_error_distribution_temporal','Observation error distribution and temporal target fidelity, not executed vibration')
# Training curves: comparable C01 latent objective only; original baseline action loss excluded.
fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
for key,rel,label in [('c01_lora97500','A800/run/pi_lora_100k','LoRA batch1'),('c01_fullft60000','A100/run/pi_fullft_dp2_100k','Full FT global batch2')]:
 rowsval=[json.loads(s) for s in (R/'raw'/rel/'validation_metrics.jsonl').read_text().splitlines()];by={x['step']:x for x in rowsval};val=[by[s] for s in sorted(by)]
 rowslearn=[json.loads(s) for s in (R/'raw'/rel/'training_metrics.jsonl').read_text().splitlines()];by={x['step']:x for x in rowslearn};learn=[by[s] for s in sorted(by)]
 for field,ax in [('loss',axs[0]),('effect_loss',axs[1])]:ax.plot([x['step']/1000 for x in val],[x[field] for x in val],color=colors[key],ls=styles[key],label=label,lw=1.5)
 # Exclude early ramp changes from a fair fixedweight composite interpretation.
axs[0].axvspan(0,10,color='#dddddd',alpha=.3);axs[0].set_title('Heldout composite loss (ramp reaches1 at10k)');axs[1].set_title('Heldout E flow loss');
for ax in axs:ax.set_xlabel('Optimizer steps (thousands)');ax.set_ylabel('C01 latent-space loss');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
save(fig,'05_c01_validation_curves','C01 validation objective; baseline action-space loss intentionally not mixed into this figure')
# Training evolution is C01-only; different batches and loss gates are disclosed.
fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
from scipy.ndimage import median_filter
for key,rel,label in [('c01_lora97500','A800/run/pi_lora_100k','LoRA batch1'),('c01_fullft60000','A100/run/pi_fullft_dp2_100k','Full FT global batch2')]:
 rowslearn=[json.loads(s) for s in (R/'raw'/rel/'training_metrics.jsonl').read_text().splitlines()];by={x['step']:x for x in rowslearn};learn=[by[s] for s in sorted(by)];x=np.array([a['step']/1000 for a in learn]);y=np.array([a['loss'] for a in learn]);smooth=median_filter(y,size=101,mode='nearest');axs[0].plot(x,smooth,color=colors[key],ls=styles[key],label=label,lw=1.4)
 with_lr=[a for a in learn if 'lr' in a]
 if with_lr:axs[1].plot([a['step']/1000 for a in with_lr],[a['lr'] for a in with_lr],color=colors[key],ls=styles[key],label=label,lw=1.4)
axs[0].set_title('C01 training loss: rolling median101 logged points');axs[0].set_ylabel('Composite latent loss');axs[1].set_title('Recorded learning rate (LoRA logging starts40k)');axs[1].set_ylabel('Learning rate')
for ax in axs:ax.set_xlabel('Optimizer steps (thousands)');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
save(fig,'07_training_loss_and_lr','C01 training evolution; unequal batch/LR protocols disclosed, not action-space baseline loss')
# Predeclared representative examples: first fixed anchor for each task, not selected by error.
selected=[]
for task_id in range(7):
 n=int(np.flatnonzero(task==task_id)[0]);selected.append({'task_index':task_id,'ordinal':n,'dataset_index':int(ref['indices'][n]),'source_episode':int(sources[n]),'subtask_episode':int(ref['subtask_episode_indices'][n]),'frame':int(ref['frame_indices'][n]),'selection':'first fixed evaluation anchor for this task; chosen independent of model error'})
 fig,axs=plt.subplots(4,4,figsize=(12,8.2),sharex=True,layout='constrained')
 for j,ax in enumerate(axs.flat):
  x=np.arange(30)/30;ax.plot(x,gt[n,:,j],color='#363636',ls=':',lw=1.8,label='Demonstration action')
  for key,(label,_) in models.items():ax.plot(x,arrays[key]['predictions'][n,:,j],color=colors[key],ls=styles[key],lw=1.3,label=label)
  
  if not mask[n].all():ax.axvspan(int(mask[n].sum())/30,29/30,color='#dddddd',alpha=.4)
  ax.scatter([0],[state[n,j]],marker='x',s=25,color='#252525',label='Observed state' if j==0 else None,zorder=5)
  ax.set_title(names[j],fontsize=9);ax.grid(axis='y',alpha=.15)
  if j>=12:ax.set_xlabel('Prediction offset (s)')
  if j%4==0:ax.set_ylabel('Motor position (native units)')
 handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False,fontsize=8)
 fig.suptitle(f'Task{task_id}: {task_names[task_id]} | same observation #{n}, source episode{sources[n]}, frame{ref["frame_indices"][n]}\nSingle action chunk: teacher observation, no rollout or robot execution',fontsize=11)
 save(fig,f'openloop_task{task_id}_anchor{n:03d}','All16 action trajectories for one identical current observation; first fixed anchor for task')
(O/'openloop_examples.json').write_text(json.dumps(selected,indent=2))
# Observation contact sheet matches the precise selected chunks.
fig,axs=plt.subplots(7,3,figsize=(9,14),layout='constrained')
for i,v in enumerate(selected):
 for j,view in enumerate(['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']):
  p=models['c01_lora97500'][1]/f'obs_task{i}_anchor{v["ordinal"]}_{view}.png';axs[i,j].imshow(Image.open(p));axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
  if i==0:axs[i,j].set_title(['Egocentric','Left wrist','Right wrist'][j])
  if j==0:axs[i,j].set_ylabel(f'Task{i}\n{task_names[i]}',fontsize=8)
fig.suptitle('Current observations used by the selected open-loop action chunks',fontsize=12);save(fig,'06_observation_contact_sheet','Three current RGB views for deterministic per-task representative anchors')
# Supplementary failure diagnostic chosen by largest FT gripper MSE, not a representative case.
err=arrays['c01_fullft60000']['predictions']-gt;severity=np.nanmean(err[:,:,14:]**2,axis=(1,2));n=int(np.argmax(severity));joints=[14,15]
fig,axs=plt.subplots(1,3,figsize=(11,3.2),layout='constrained');x=np.arange(30)/30
for ax,j in zip(axs[:2],joints):
 ax.plot(x,gt[n,:,j],color='#303030',ls=':',lw=1.8,label='Demonstration action');ax.plot(x,ref['future_states'][n,:,j],color='#c4c1bc',ls='--',lw=1,label='Measured q (demo)')
 for key,(label,_) in models.items():ax.plot(x,arrays[key]['predictions'][n,:,j],color=colors[key],ls=styles[key],label=label,lw=1.5)
 ax.set_title(names[j]);ax.set_xlabel('Prediction offset (s)');ax.set_ylabel('Motor position (native units)')
for key,(label,_) in models.items():axs[2].plot(x,np.nanmean(np.abs(arrays[key]['predictions'][n,:,14:]-gt[n,:,14:]),axis=-1),color=colors[key],ls=styles[key],label=label)
axs[2].set_title('Mean absolute gripper error');axs[2].set_xlabel('Prediction offset (s)');axs[2].set_ylabel('Error (native units)');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False,fontsize=8)
fig.suptitle(f'Worst FT gripper-MSE diagnostic: observation#{n}, task{task[n]}, source episode{sources[n]}\nSelected by FT error; not representative of all200 observations',fontsize=10)
save(fig,'08_worst_fullft_gripper_diagnostic','Largest FT gripper MSE observation, explicit diagnostic selection; teacher-observation target and measured demo state')
(O/'worst_gripper_example.json').write_text(json.dumps({'ordinal':n,'dataset_index':int(ref['indices'][n]),'task_index':int(task[n]),'source_episode':int(sources[n]),'criterion':'maximum fullFT60000 per-observation gripper mean squared action error over valid steps','ft_gripper_rmse':float(np.sqrt(severity[n]))},indent=2))
fig,axs=plt.subplots(1,2,figsize=(8.4,3.2),layout='constrained')
for key,(label,_) in models.items():
 hs=[1,5,8,10,16,20,30];m=result['models'][key]['prefix_windows'];axs[0].plot(hs,[m[str(h)]['mae'] for h in hs],color=colors[key],ls=styles[key],marker='o',ms=3,label=label);axs[1].plot(hs,[m[str(h)]['rmse'] for h in hs],color=colors[key],ls=styles[key],marker='o',ms=3,label=label)
for ax,title in zip(axs,['Action MAE','Action RMSE']):ax.set_title(title);ax.set_xlabel('Prefix length (steps)');ax.set_ylabel('Error (native units)');ax.set_xticks([1,8,16,30]);ax.grid(axis='y',alpha=.15)
axs[0].legend(frameon=False);save(fig,'09_prefix_window_accuracy','Accuracy of firstN predicted targets; offline diagnostic, no control-window tuning performed')
pdf_all.close()
(F/'figure_catalog.json').write_text(json.dumps(catalog,indent=2));print(json.dumps({k:{f:v[f] for f in ['mae','rmse','range_normalized_mae']} for k,v in result['models'].items()},indent=2));print('OUTPUT',O)
