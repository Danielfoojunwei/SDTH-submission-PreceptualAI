"""Plot genuine RT renders, DINO feature PCA and same-scene radio couplings."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    receipt_path=ROOT/'artifacts/visual_radio/rt_probe/receipt.json'
    receipt=json.loads(receipt_path.read_text());assert receipt['geometry_radio_validation_passed']
    data=Path(receipt['dinov3']['feature_path']);assert sha(data)==receipt['dinov3']['feature_sha256']
    with np.load(data,allow_pickle=False) as archive:
        tokens=archive['patch_tokens'].astype('f8');boxes=archive['patch_boxes_original_pixels']
    assert tokens.shape==(2,256,384) and np.isfinite(tokens).all()
    flat=tokens.reshape(-1,384);centered=flat-flat.mean(0)
    _,_,axes=np.linalg.svd(centered,full_matrices=False)
    rgb=centered@axes[:3].T;low,high=np.percentile(rgb,[2,98],axis=0)
    rgb=np.clip((rgb-low)/np.maximum(high-low,1e-12),0,1).reshape(2,16,16,3)
    gains=np.array([p['incoherent_path_power_rx_tx'] for p in receipt['poses']])
    assert gains.shape==(2,2,2) and (gains>0).all()
    db=10*np.log10(gains)
    fig,ax=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for index,pose in enumerate(receipt['poses']):
        assert sha(pose['rgb_path'])==pose['rgb_sha256']
        ax[index,0].imshow(Image.open(pose['rgb_path']));ax[index,0].axis('off')
        x0,y0=boxes[index,:,:2].min(0);x1,y1=boxes[index,:,2:].max(0)
        ax[index,0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='white',linestyle='--',linewidth=1))
        ax[index,0].set_title('RT render · blocker '+str(tuple(pose['blocker_position_m'])),fontsize=10)
        ax[index,1].imshow(rgb[index],interpolation='nearest');ax[index,1].axis('off')
        ax[index,1].set_title('DINOv3 patches · shared 3-component PCA',fontsize=10)
        heat=ax[index,2].imshow(db[index],vmin=db.min(),vmax=db.max(),cmap='viridis')
        ax[index,2].set(xticks=[0,1],yticks=[0,1],xticklabels=['TX0','TX1'],yticklabels=['RX0','RX1'],
                       title='Ray-traced path coupling [dB]')
        for r in range(2):
            for t in range(2):ax[index,2].text(t,r,f'{db[index,r,t]:.2f}',ha='center',va='center',color='white' if db[index,r,t]<db.mean() else 'black',fontsize=12)
    fig.colorbar(heat,ax=ax[:,2],shrink=.8,label='10 log10(sum |a_path|²)')
    fig.suptitle('Visible geometry and RF coupling are paired; prediction quality remains untested',fontsize=14)
    output=ROOT/'docs/figures/dinov3_physical_radio.png';output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160);plt.close(fig)
    record={'rt_receipt_sha256':sha(receipt_path),'feature_array_sha256':sha(data),
        'figure_path':str(output.relative_to(ROOT)),'figure_sha256':sha(output),
        'script_sha256':sha(Path(__file__)),
        'interpretation':'Two static simulated poses. Dashed rectangle shows the actual DINO crop. Joint PCA gives descriptive colors, not object labels, depth, radio predictions, transfer or collapse validation. Renders use four samples/pixel; visual noise remains. RF values are path couplings, not packet performance.'}
    (ROOT/'artifacts/visual_radio/figure.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
