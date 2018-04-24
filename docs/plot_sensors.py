import mne
import numpy as np
from mne.datasets import sample
from mayavi import mlab
import matplotlib.pyplot as plt
from mne.viz._3d import _create_mesh_surf


# read average plot

data_path = sample.data_path() + '/MEG/sample/'
evoked = mne.read_evokeds(data_path + 'sample_audvis-ave.fif')[-1]
#fsaverage_audvis-meg-rh.stc

# plot brain
trans_fname = None
trans_fname = data_path + 'sample_audvis_raw-trans.fif'
subjects_dir = '/home/jrking/freesurfer/subjects/'
subjects_dir = sample.data_path() + '/subjects/'
mne.viz.plot_alignment(evoked.info, trans=trans_fname, subject='sample',
                       eeg=False, meg='sensors',
                       subjects_dir=subjects_dir, surfaces=('brain', 'head'))
fig = mlab.gcf()
fig.scene.background = (1., 1., 1.)

# plot scalp
evoked.pick_types(eeg=True, meg=False)
this_map = mne.make_field_map(evoked, trans=trans_fname, subject='sample',
                              subjects_dir=subjects_dir, n_jobs=1)[0]


# plot field
time = 0.096
time_idx = np.argmin(np.abs(evoked.times - time))
surf = this_map['surf']
map_data = this_map['data']
data = np.dot(map_data, evoked.data[:, time_idx])
vlim = np.percentile(np.abs(data), 90)
mesh = _create_mesh_surf(surf, fig)
colormap = plt.get_cmap('RdBu_r')(np.linspace(.1, .9, 255)) * 255
alphas = np.linspace(.9, .1, 128) * 255
colormap[:, -1] = np.r_[alphas, alphas[:0:-1]]
mesh = _create_mesh_surf(surf, fig, scalars=data)
fsurf = mlab.pipeline.surface(mesh, vmin=-vlim, vmax=vlim, figure=fig)
fsurf.module_manager.scalar_lut_manager.lut.table = colormap.astype(int)
fsurf.actor.property.backface_culling = True

fig = mlab.gcf()
fig.scene.background = (1., 1., 1.)

mlab.view(-162.00694177264324, 79.12950725401194, 0.48815080523496035)
mlab.draw()
