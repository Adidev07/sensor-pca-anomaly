from common import *
def normal(rng,n):
    z=rng.normal(size=(n,2));a=np.array([[1,.2],[.8,-.2],[.3,1],[-.2,.9],[.7,.6]])
    return z@a.T+rng.normal(0,.08,(n,5))
def fit(x,k=2):
    mean=x.mean(axis=0);std=x.std(axis=0);std=np.where(std<1e-12,1,std)
    _,_,vt=np.linalg.svd((x-mean)/std,full_matrices=False)
    return dict(mean=mean,std=std,components=vt[:k])
def score(model,x):
    z=(x-model['mean'])/model['std'];reconstructed=(z@model['components'].T)@model['components']
    return np.mean((z-reconstructed)**2,axis=1)
def main():
    rng=np.random.default_rng(42);train=normal(rng,700);val=normal(rng,250);test=normal(rng,400);y=np.zeros(400,dtype=int)
    selected=rng.choice(400,60,replace=False);test[selected,3]+=rng.uniform(2,5,len(selected));y[selected]=1
    m=fit(train);threshold=float(np.quantile(score(m,val),.99));s=score(m,test)
    report=dict(data='700 normal training, 250 normal validation, 400 test rows including 60 injected anomalies; five synthetic sensors',threshold=threshold,test=binary_metrics(y,s>threshold),always_normal_baseline=binary_metrics(y,np.zeros_like(y)),score_summary=dict(normal_mean=float(s[y==0].mean()),anomaly_mean=float(s[y==1].mean())),caveat='Detects deviations outside the learned subspace. In-subspace anomalies and drift can be missed; injected spikes are deliberately simple.')
    out=ROOT/'results';out.mkdir(exist_ok=True);np.savez(out/'model.npz',**m,threshold=threshold)
    np.savetxt(out/'test_scores.csv',np.column_stack([test,y,s]),delimiter=',',header='s1,s2,s3,s4,s5,is_anomaly,reconstruction_error',comments='')
    save_report(report)
if __name__=='__main__': main()
