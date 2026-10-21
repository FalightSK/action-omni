from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P=Path(__file__).resolve().parent
def data(name):
    path=P/'figure_data'/name
    return list(csv.DictReader(path.open(encoding='utf8')))
rows=data('aloha_measured.csv')
final=data('libero_final.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
colors={'groot':'#2166ac','qwen3vl':'#d66026'}
fig,axes=plt.subplots(1,3,figsize=(7.0,2.0),sharex=True,sharey=True)
for ax,seed in zip(axes,['3','4','5']):
    for backbone in ['groot','qwen3vl']:
        rr=sorted([r for r in rows if r['training_seed']==seed and r['backbone']==backbone],key=lambda r:int(r['epoch']))
        if rr:
            ax.plot([int(r['epoch']) for r in rr],[100*float(r['success_rate']) for r in rr],color=colors[backbone],marker='o' if backbone=='groot' else 's',ms=3.5,lw=1.0)
        elif backbone=='qwen3vl':
            ax.text(.5,.86,'Qwen: no measured checkpoints',ha='center',transform=ax.transAxes,color=colors[backbone],fontsize=7)
    ax.set(title=f'Training seed {seed}',xlim=(40,260),ylim=(0,100),xticks=[50,100,150,200,250],yticks=[0,25,50,75,100],xlabel='Training epoch')
    ax.grid(axis='y',alpha=.22,lw=.6);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Development success (%)')
legend=[Line2D([0],[0],color=colors['groot'],marker='o',lw=1,label='GR00T'),Line2D([0],[0],color=colors['qwen3vl'],marker='s',lw=1,label='Qwen')]
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,fontsize=8)
fig.tight_layout(rect=(0,0,1,.88),pad=.4,w_pad=1.0)
fig.savefig(P/'figures/aloha_development.pdf',bbox_inches='tight');fig.savefig(P/'figures/aloha_development.png',dpi=200,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(3.4,1.55))
for y,r in enumerate(final):
    rate=100*float(r['success_rate']);c=colors['groot' if r['backbone']=='GR00T' else 'qwen3vl']
    ax.barh(y,rate,height=.5,color=c,alpha=.9)
    ax.text(rate-2,y,f"{rate:.2f}% ({r['successes']}/{r['episodes']})",va='center',ha='right',fontsize=7,color='white',fontweight='bold')
ax.set(yticks=range(len(final)),yticklabels=[f"{r['backbone']} seed {r['training_seed']}" for r in final],xlim=(0,100),xticks=[0,25,50,75,100],xlabel='Held-out final success (%)')
ax.invert_yaxis();ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
fig.tight_layout(pad=.4);fig.savefig(P/'figures/libero_final.pdf',bbox_inches='tight');fig.savefig(P/'figures/libero_final.png',dpi=200,bbox_inches='tight');plt.close(fig)
print(f'Plotted {len(rows)} measured ALOHA checkpoints and {len(final)} final LIBERO policies; no imputation.')
