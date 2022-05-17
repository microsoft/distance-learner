import os 
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')

from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .common import *

def make_plots(model, dataset, dump_dir, task="regression",\
     batch_size=512, num_workers=8, cuda=0, thresh=None, num_points=50000):

    # intertwined swiss rolls are planar swiss rolls
    # so we will proceed as such
    n = 0
    k = 0
    if isinstance(dataset, manifold.Manifold):
        n = dataset.genattrs.n
        k = dataset.genattrs.k
        if thresh is None:
            thresh = dataset.genattrs.D / dataset.genattrs.norm_factor
    else:
        n = dataset.n
        k = dataset.k
        if thresh is None:
            thresh = dataset.S1.genattrs.D / dataset.norm_factor

    points_k = get_coplanar_kdim_samples(dataset)
    points_k_classes = dataset.class_labels
    gen_kd_grid, gen_nd_grid = get_nplane_samples_for_kmfld(points_k, dataset, n, num_points)
    num_classes = model.output_size 
    dummy_labels = torch.from_numpy(np.zeros((num_points, num_classes))).float()

    if task == "clf":
        dummy_labels = dummy_labels[:, 0].long()

    gen_nd_dataset = TensorDataset(gen_nd_grid, dummy_labels)

    gen_nd_dl = DataLoader(dataset=gen_nd_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() and cuda is not None else "cpu")

    _, _, _, gen_nd_logits = lmd.test(model, gen_nd_dl, device, task=task, debug=False)

    gen_pred_classes = None
    if task == "clf":
        gen_pred_classes = torch.max(gen_nd_logits, axis=1)[1]
    elif task == "regression":
        gen_pred_classes = torch.min(gen_nd_logits, axis=1)[1]


    plot_decision_regions(gen_kd_grid, points_k, points_k_classes, gen_nd_grid,\
         gen_pred_classes, gen_nd_logits, dataset, thresh, task, dump_dir)


def plot_decision_regions(gen_kd_grid, points_k, points_k_classes,\
     gen_nd_grid, gen_pred_classes, gen_nd_logits, dataset, thresh, task, dump_dir):
    print(gen_kd_grid.shape)
    plotdir = os.path.join(dump_dir, "analysis_plots")
    os.makedirs(plotdir, exist_ok=True)

    THRESH = thresh
    OFF_MFLD_LABEL = torch.max(gen_pred_classes) + 1

    k = points_k.shape[1]
    n = gen_nd_grid.shape[1]
    if k not in [2, 3]:
        raise RuntimeError("decision region visualization not possible")

    if task == "regression": gen_pred_classes[torch.min(gen_nd_logits, axis=1)[0] >= THRESH] = OFF_MFLD_LABEL
    col = ["blue", "green", "red"]

    if k == 2:

        plt.figure(figsize=(6, 6))

        for i in range(len(col)):
            plt.scatter(gen_kd_grid[gen_pred_classes.numpy() == i, 0].numpy(), gen_kd_grid[gen_pred_classes.numpy() == i, 1].numpy(), s=0.01, c=col[i], label=i)
            if (i < 2):
                plt.scatter(points_k[points_k_classes == i, 0], points_k[points_k_classes == i, 1], c=col[i], s=0.1)

        plt.legend(markerscale=100)
        plt.title("fig.1: clf labels with off-manifold label (2) for {}".format("dist regressor" if task == "regression" else "stdclf"))
        plt.savefig(os.path.join(plotdir, "fig-1.png"))

        if n == 3:

            data = list()

            num_classes = np.unique(dataset.class_labels)[-1] + 1

            for i in range(num_classes):
                # creating Scatter3D objects for the grid
                data.append(
                    go.Scatter3d(
                        x=gen_nd_grid[:, 0][gen_pred_classes==i],
                        y=gen_nd_grid[:, 1][gen_pred_classes==i],
                        z=gen_nd_grid[:, 2][gen_pred_classes==i],
                        name="S{}".format(i+1) if i < num_classes - 1 else "off-mfld",
                        mode='markers',
                        marker=dict(
                            size=1,
                            color=col[i],                # set color to an array/list of desired values
                            colorscale='Viridis',   # choose a colorscale
                            opacity=0.8
                        )
                    )
                )

                # creating Scatter3d points for the on-mfld. samples of the data
                if i < num_classes - 1:
                    data.append(
                        go.Scatter3d(
                            x=dataset.normed_all_points[:, 0][dataset.class_labels==i],
                            y=dataset.normed_all_points[:, 1][dataset.class_labels==i],
                            z=dataset.normed_all_points[:, 2][dataset.class_labels==i],
                            name="S{}".format(i+1) if i < num_classes - 1 else "off-mfld",
                            mode='markers',
                            marker=dict(
                                size=2,
                                color=col[i],                # set color to an array/list of desired values
                                colorscale='Viridis',   # choose a colorscale
                                opacity=0.8
                            ),
                            showlegend=False
                        )
                    )

            fig = go.Figure(data=data)
            
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

            fig.write_html(os.path.join(plotdir, "fig-1-3D.html"))


    elif k == 3:

        data = list()

        num_classes = np.unique(dataset.class_labels)[-1] + 1

        for i in range(num_classes):
            # creating Scatter3D objects for the grid
            data.append(
                go.Scatter3d(
                    x=gen_kd_grid[:, 0][gen_pred_classes==i],
                    y=gen_kd_grid[:, 1][gen_pred_classes==i],
                    z=gen_kd_grid[:, 2][gen_pred_classes==i],
                    name="S{}".format(i+1) if i < num_classes - 1 else "off-mfld",
                    mode='markers',
                    marker=dict(
                        size=1 if i == num_classes - 1 else 2,
                        color=col[i],                # set color to an array/list of desired values
                        colorscale='Viridis',   # choose a colorscale
                        opacity=0.9 if i < num_classes - 1 else 0.2
                    )
                )
            )

            # creating Scatter3d points for the on-mfld. samples of the data
            if i < num_classes - 1:
                data.append(
                    go.Scatter3d(
                        x=dataset.normed_all_points[:, 0][dataset.class_labels==i],
                        y=dataset.normed_all_points[:, 1][dataset.class_labels==i],
                        z=dataset.normed_all_points[:, 2][dataset.class_labels==i],
                        name="S{}".format(i+1) if i < num_classes - 1 else "off-mfld",
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=col[i],                # set color to an array/list of desired values
                            colorscale='Viridis',   # choose a colorscale
                            line=dict(width=1,
                                        color='DarkSlateGrey'),
                            opacity=0.9
                        ),
                        showlegend=False
                    )
                )

        fig = go.Figure(data=data)
        
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

        fig.write_html(os.path.join(plotdir, "fig-1.html")) 

        fig2 = go.Figure(data=data[:-1])
        
        fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0))

        fig2.write_html(os.path.join(plotdir, "fig-1-on-mfld.html"))    

    
def plot_dist_hmaps():
    pass

    



    

    



            