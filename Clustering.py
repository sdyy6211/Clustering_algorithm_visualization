import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull
import random
import io
import PIL

colors = ['red','green','purple','blue','orange','pink']

def kmeans(data,k,eps,init_num = 10):
    
    centroids_list = np.zeros((init_num,k,data.shape[1]))
    
    loss_list = np.zeros((init_num,1))
    
    fig_list = []
    
    for each_run in range(init_num):
        
        fig_ = []
        fig, ax = plt.subplots(figsize = (16,9),dpi = 100)
        ax.set_xlim(min(data[:,0])-abs(min(data[:,0])*0.1),max(data[:,0])+abs(max(data[:,0])*0.1))
        ax.set_ylim(min(data[:,1])-abs(min(data[:,1])*0.1),max(data[:,1])+abs(max(data[:,1])*0.1))
        plt.tight_layout()
        ax.set_title('loss: -')
        plt.close()
        
        ax.scatter(data[:,0],data[:,1],alpha = 0.5)
        
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        img_buf.detach

        fig_.append(img_buf)
    
        labels = np.zeros((len(data),1))

        centroids = kmeans_initialization(data, k)
        
        for i in range(k):
            
            ax.scatter(centroids[i,0],centroids[i,1],alpha = 0.5,marker = '*',color = colors[i],s = 100)
            
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        img_buf.detach

        fig_.append(img_buf)
        
        max_num = 100
        
        count = 0

        while True:

            labels = assign_points(data,centroids)
            
            for i in range(k):
                
                ax.scatter(data[labels==i,0],data[labels==i,1],alpha = 0.5,color = colors[i])

            centroids_previous = centroids

            centroids = calculate_center(data, labels, k, centroids)
            
            for i in range(k):
                
                ax.scatter(centroids[i,0],centroids[i,1],alpha = 0.5,marker = '*',color = colors[i],s = 100)
                
                lenx = centroids[i,0]-centroids_previous[i,0]+1e-8
                leny = centroids[i,1]-centroids_previous[i,1]+1e-8

                ax.arrow(centroids_previous[i,0],centroids_previous[i,1],lenx,leny, length_includes_head=True,head_width=0.1, head_length = 0.1,alpha = 0.5)
                
            current_loss,previous_loss = getloss(data, labels, k, centroids, centroids_previous)
            
            ax.set_title('loss: {0}'.format(current_loss))
            
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png')
            img_buf.detach
            
            fig_.append(img_buf)

            if (previous_loss-current_loss < eps) or (count >= max_num):

                break
                
            count+=1
                
        centroids_list[each_run,:,:] = centroids
        
        loss_list[each_run] = current_loss
        
        fig_list.append(fig_)
        
    loss_list[np.isnan(loss_list)] = np.inf
        
    return centroids_list[loss_list.argmin()],assign_points(data,centroids_list[loss_list.argmin()]),[PIL.Image.open(f) for f in fig_list[loss_list.argmin()]]
    
def kmeans_initialization(data, k):
    
    centroids = np.array(random.choices(data,k = k))
    
    return centroids
    
def assign_points(data, centroids):
    
    outer_result = np.linalg.norm((data[:,None,:] - centroids),axis = 2)
    
    return outer_result.argmin(axis = 1)
    
def calculate_center(data, labels, k, centroids):
    
    dummies = np.zeros((labels.shape[0],k))
    
    dummies[np.arange(labels.size),labels] = 1
    
    dummies = dummies/dummies.sum(axis = 0)
    
    centroids = dummies.T@data
    
    return centroids
    
def getloss(data, labels, k, centroids, centroids_previous):
    
    dummies = np.zeros((labels.shape[0],k))
    
    dummies[np.arange(labels.size),labels] = 1
    
    current_loss = (np.linalg.norm((data[:,None,:] - centroids),axis = 2)*dummies).sum()
    
    previous_loss = (np.linalg.norm((data[:,None,:] - centroids_previous),axis = 2)*dummies).sum()

    return current_loss,previous_loss


###################################################################################################################
def cov_to_ellipse(cov):
    r1 = (cov[0,0]+cov[1,1])/2+np.sqrt(((cov[0,0]-cov[1,1])/2)**2+cov[0,1]**2)
    r2 = (cov[0,0]+cov[1,1])/2-np.sqrt(((cov[0,0]-cov[1,1])/2)**2+cov[0,1]**2)
    
    if cov[0,1] == 0 and cov[0,0]>=cov[1,1]:
        
        theta = 0
        
    if cov[0,1] == 0 and cov[0,0]<cov[1,1]:
        
        theta = np.pi/2
        
    else:
        
        theta = np.arctan2(r1 - cov[0,0], cov[0,1])
        
    return np.sqrt(r1)*2,np.sqrt(r2)*2,theta*180/np.pi
    
def mvgaussian(x,mu,sigma):
    
    D = x.shape[1]
    
    normalizer = 1/((2*np.pi)**(D/2)*np.linalg.det(sigma)**(1/2))
    
    unormalized_density = np.exp(-1/2*(x[:,None,:]-mu)@np.linalg.inv(sigma)@(x[:,None,:]-mu).transpose(0,2,1))
    
    return (normalizer*unormalized_density).squeeze()

def gmm_initialization(X,K):
    
    N = X.shape[0]
    
    D = X.shape[1]

    Mu = np.zeros((K,D))

    Sigma = np.zeros((K,D,D))

    Phi = np.zeros((K,1))

    for i in range(K):

        Mu[i,:] = np.array(random.choices(X,k = 1)).reshape(-1,)

        Sigma[i,:,:] = np.eye(D)

        Phi[i] = 1/K
        
    W = np.zeros((N,K))
        
    return Mu,Sigma,Phi,W

def E_step(X,Mu,Sigma,Phi,W):
    
    K = len(Mu)
    
    N = X.shape[0]
    
    for i in range(K):
        
        W[:,i] = mvgaussian(X,Mu[i,:],Sigma[i,:,:])*Phi[i]/sum(mvgaussian(X,Mu[i,:],Sigma[i,:,:])*Phi[i] for i in range(K))
    
    return W

def M_step(X,W,Mu,Sigma,Phi):
    
    K = len(Mu)
    
    N = X.shape[0]
    
    for i in range(K):
    
        w = W[:,i]

        mu = Mu[i,:]

        sigma = Sigma[i,:,:]

        phi = Phi[i]

        mu_update = (w.reshape(-1,1) * X).sum(axis = 0)/w.sum(axis = 0)

        sigma_update = ((w.reshape(-1,1)*(X-mu))/w.reshape(-1,1).sum(axis = 0)).T@(X-mu)

        phi_update = w.sum(axis = 0)/N

        Mu[i] = mu_update

        Sigma[i] = sigma_update

        Phi[i] = phi_update
        
    return Mu,Sigma,Phi

def calculate_negative_likelihood(W,X,Mu,Sigma,Phi):
    K = len(Mu)
    
    l = 0
    
    for i in range(K):
        
        l += (W[:,i]*np.log(mvgaussian(X,Mu[i,:],Sigma[i,:,:])*Phi[i]/W[:,i])).sum()
        
    return -l

def GMM(X,K,eps = 1e-3, max_iter = 100):
    
    fig_list = []
    fig, ax = plt.subplots(figsize = (16,9),dpi = 100)
    ax.set_xlim(min(X[:,0])-abs(min(X[:,0])*0.1),max(X[:,0])+abs(max(X[:,0])*0.1))
    ax.set_ylim(min(X[:,1])-abs(min(X[:,1])*0.1),max(X[:,1])+abs(max(X[:,1])*0.1))
    ax.set_title('loss: -')
    plt.tight_layout()
    plt.close()
    ax.scatter(X[:,0],X[:,1],alpha = 0.5)
    
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.detach

    fig_list.append(img_buf)
    
    
    Mu,Sigma,Phi,W = gmm_initialization(X,K)
    l_previous = np.inf
    Mu_prev = Mu.copy()
    
    counter = 0

    while True:
        
        fig, ax = plt.subplots(figsize = (16,9),dpi = 100)
        ax.set_xlim(min(X[:,0])-abs(min(X[:,0])*0.1),max(X[:,0])+abs(max(X[:,0])*0.1))
        ax.set_ylim(min(X[:,1])-abs(min(X[:,1])*0.1),max(X[:,1])+abs(max(X[:,1])*0.1))
        plt.tight_layout()
        plt.close()

        W = E_step(X,Mu,Sigma,Phi,W)

        Mu,Sigma,Phi = M_step(X,W,Mu,Sigma,Phi)

        l = calculate_negative_likelihood(W,X,Mu,Sigma,Phi)

        #print(l)

        if (l_previous - l < eps) or (counter >= max_iter) or (np.isinf(l)):

            break

        l_previous = l
        
        counter+=1
        
        for c,(m,s) in enumerate(zip(Mu,Sigma)):
            r1,r2,theta = cov_to_ellipse(s)
            assign_index = np.where(W.argmax(axis = 1)==c)[0]
            
            lenx = m[0]-Mu_prev[c][0]+1e-8
            leny = m[1]-Mu_prev[c][1]+1e-8
            
            ax.scatter(m[0],m[1],color = colors[c],alpha = 0.5,marker = '*',s = 100)
            ax.scatter(X[assign_index,0],X[assign_index,1],color = colors[c],alpha = 0.5)
            ax.add_patch(Ellipse((m[0],m[1]),r1,r2,angle = theta,fill = False,color = colors[c]))
            ax.arrow(Mu_prev[c][0],Mu_prev[c][1],lenx,leny, length_includes_head=True,head_width=0.1, head_length = 0.1,alpha = 0.5)
        
        
        ax.set_title('likelihood: {0}'.format(l))
        
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        img_buf.detach

        fig_list.append(img_buf)
        
        Mu_prev = Mu.copy()
        
    return Mu,Sigma,Phi,W,[PIL.Image.open(f) for f in fig_list]


###################################################################################################################

class Cluster:
    
    def __init__(self, linkage = 'mean'):
        
        self.datapoints = []
        
        self.index = []
        
        self.linkage = linkage
        
        self.childrean = None
        
        self.distance = 0
        
    def load(self,datapoint):
        
        for i in datapoint:
            
            self.datapoints.append(i)
            
            self.index.append(list(i.keys())[0])
        
        self.update_representation()
        
        return self
        
    def update_representation(self):
        
        self.representation = np.array([list(i.values())[0] for i in self.datapoints])
        
    def get_flat_index(self):
        
        return [list(i.keys())[0] for i in self.datapoints]

def UPGMA(x,y,distance_measurement = 'euclidean'):
    
    x = x.reshape(-1,2)
    
    y = y.reshape(-1,2)
    
    if distance_measurement == 'euclidean':
    
        dis = np.linalg.norm(x[:,None,:] - y,axis = 2).sum()
        
    else:
        
        raise(NotImplementedError)
        
    return dis/(x.shape[0]*y.shape[0])

def calculate_intra_dis(C,method = UPGMA, distance_measurement = 'euclidean'):
    
    Clen = len(C)
    
    dism = np.full((Clen,Clen),np.inf)
    
    for i in range(Clen):
        
        for j in range(i+1,Clen):
            
            dism[i,j] = method(C[i].representation,C[j].representation,distance_measurement = distance_measurement)
            
    dism = dism.T

    return dism

def merge(left,right):
    new_c = Cluster()
    new_c.childrean = [left,right]
    new_c.datapoints = left.datapoints+right.datapoints
    new_c.index = [left.index+right.index]
    new_c.update_representation()
    return new_c

def merge_cluster(clusters,
                  distance,
                  index1,
                  index2):
    
    new_clu = merge(clusters[index1],clusters[index2])
    new_clu.distance = distance
    
    del clusters[index1]
    del clusters[index2]
    
    clusters.append(new_clu)

def find_nearest_neighbour(C,method = UPGMA, distance_measurement = 'euclidean'):
    
    distances = calculate_intra_dis(C,method = method, distance_measurement = distance_measurement)
    
    distances_index = np.tril_indices_from(distances,k = -1)
    
    a1 = distances_index[0][distances[distances_index].argmin()]
    
    a2 = distances_index[1][distances[distances_index].argmin()]
    
    return a1,a2,distances[a1,a2]

def hierarchical_clustering(X,method = UPGMA, distance_measurement = 'euclidean'):
    
    N = X.shape[0]
    
    C = []

    Clusters_list = dict()

    steps = dict()

    for i in range(N):
        
        C.append(Cluster().load([{i:X[i,:]}]))
        
    fig_list = []
        
    counter = 0

    while len(C) >1:
        
        fig,ax = plt.subplots(figsize = (10,9),dpi = 100)

        ax.set_xlim(min(X[:,0])-abs(min(X[:,0])*0.1),max(X[:,0])+abs(max(X[:,0])*0.1))
        ax.set_ylim(min(X[:,1])-abs(min(X[:,1])*0.1),max(X[:,1])+abs(max(X[:,1])*0.1))

        plt.tight_layout()
        
        ax.set_title('merge: {0}'.format(str(counter+1)))
        
        ax.scatter(X[:,0],X[:,1],color = 'blue',s = 200,alpha = 0.3)
        
        a1,a2,dis = find_nearest_neighbour(C,method = method, distance_measurement = distance_measurement)
        
        Clusters_list[N-counter] = [c for c in C]
        
        steps[counter] = {'left_distance':C[a1].distance,
                        'left_index':C[a1].get_flat_index(),
                        'right_distance':C[a2].distance,
                        'right_index':C[a2].get_flat_index(),
                        'distance':dis}

        for ck,k in enumerate([a1,a2]):
            for i in C[k].datapoints:
                dp = list(i.values())[0]
                ax.scatter(dp[0],dp[1],color = colors[ck],s = 200)

        for cg in C:
            all_points = []
            for i in cg.datapoints:
                dp = list(i.values())[0]
                all_points.append(dp)
            X_ = np.array(all_points)
            if len(X_) >= 3:
                hull = ConvexHull(np.array(all_points))
                hull_index = hull.vertices.tolist()+[hull.vertices[0]]
                ax.plot(X_[hull_index,0],X_[hull_index,1],color = 'black',alpha = 0.75,linewidth = 3)

            elif len(X_) == 2:
                ax.plot(X_[:,0],X_[:,1],color = 'black',alpha = 0.75,linewidth = 3)
        
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        img_buf.detach

        fig_list.append(img_buf)
        
        merge_cluster(C,
                    dis,
                    a1,
                    a2)
        
        counter+=1

        plt.close()

    Clusters_list[N-counter] = [c for c in C]

    return Clusters_list,steps,[PIL.Image.open(f) for f in fig_list]

def plot_dendrogram(steps,C):

    fig_list = []

    fig,ax = plt.subplots(figsize = (10,9),dpi = 100)

    plt.tight_layout()
        
    mapping_table = dict(zip([list(i.keys())[0] for i in C[1][0].datapoints],range(len(steps)+1)))

    plt.ylim(-0.1,steps[len(steps)-1]['distance']*1.05)
    
    x_plot = np.array(list(mapping_table.keys()))
    
    y_plot = np.zeros((len(steps)+1,))
    
    plt.scatter(x_plot,y_plot,color = 'blue',s = 20)
    
    plt.xticks(list(mapping_table.values()),list(mapping_table.keys()),rotation = -45,fontsize = 5)

    #img_buf = io.BytesIO()
    #fig.savefig(img_buf, format='png')
    #img_buf.detach

    #fig_list.append(img_buf)

    for i in range(0,len(steps)):

        left_index = [mapping_table[index] for index in steps[i]['left_index']]

        right_index = [mapping_table[index] for index in steps[i]['right_index']]

        all_index = left_index+right_index

        x_coor = np.mean(all_index)
        y_coor = steps[i]['distance']

        plt.text(x_coor,y_coor*1.01,i+1,color = 'blue',fontsize = 8)

        point1 = np.array(
            (np.mean(left_index),steps[i]['left_distance'])
        )

        point2 = np.array(
            (np.mean(right_index),steps[i]['right_distance'])
        )

        target = np.array(
            (np.mean(all_index),steps[i]['distance'])
        )

        lenxleft = target[0]-point1[0]+1e-8
        lenyleft = target[1]-point1[1]+1e-8

        lenxright = target[0]-point2[0]+1e-8
        lenyright = target[1]-point2[1]+1e-8

        plt.arrow(point1[0],point1[1],0,lenyleft, length_includes_head=False,head_width=0, head_length =0,alpha = 0.75)
        plt.arrow(point2[0],point2[1],0,lenyright, length_includes_head=False,head_width=0, head_length = 0,alpha = 0.75)

        plt.arrow(point1[0],point1[1]+lenyleft,lenxleft,0, length_includes_head=False,head_width=0, head_length =0,alpha = 0.75)
        plt.arrow(point2[0],point2[1]+lenyright,lenxright,0, length_includes_head=False,head_width=0, head_length = 0,alpha = 0.75)

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        img_buf.detach
        
        fig_list.append(img_buf)

    plt.close()
        
    return [PIL.Image.open(f) for f in fig_list]

def Hierarchical_clustering(X,method = UPGMA, distance_measurement = 'euclidean'):
    Clusters_list,steps,figure = hierarchical_clustering(X,method = method, distance_measurement = distance_measurement)
    dfig = plot_dendrogram(steps,Clusters_list)

    total_imgs = []
    for i,j in zip(figure,dfig):
        
        img_ = PIL.Image.new('RGB', (2000, 900))
        img_.paste(j, (0,0))
        img_.paste(i, (1000,0))
        total_imgs.append(img_)

    return Clusters_list,total_imgs

###################################################################################################################

def MyDBSCAN(X, eps, MinPts):
    """
    Cluster the dataset `D` using the DBSCAN algorithm.
    
    MyDBSCAN takes a dataset `D` (a list of vectors), a threshold distance
    `eps`, and a required number of points `MinPts`.
    
    It will return a list of cluster labels. The label -1 means noise, and then
    the clusters are numbered starting from 1.
    """
 
    # This list will hold the final cluster assignment for each point in D.
    # There are two reserved values:
    #    -1 - Indicates a noise point
    #     0 - Means the point hasn't been considered yet.
    # Initially all labels are 0.    
    
    fig_list = []
    fig, ax = plt.subplots(figsize = (16,9),dpi = 100)
    ax.set_xlim(min(X[:,0])-abs(min(X[:,0])*0.1),max(X[:,0])+abs(max(X[:,0])*0.1))
    ax.set_ylim(min(X[:,1])-abs(min(X[:,1])*0.1),max(X[:,1])+abs(max(X[:,1])*0.1))
    plt.tight_layout()
    plt.close()
    ax.scatter(X[:,0],X[:,1],alpha = 0.5)
    
    labels = [0]*len(X)

    # C is the ID of the current cluster.    
    C = 0
    
    # This outer loop is just responsible for picking new seed points--a point
    # from which to grow a new cluster.
    # Once a valid seed point is found, a new cluster is created, and the 
    # cluster growth is all handled by the 'expandCluster' routine.
    
    # For each point P in the Dataset D...
    # ('P' is the index of the datapoint, rather than the datapoint itself.)
    P_ = -1
    for P in range(0, len(X)):
        ax.text(X[P,0],X[P,1],P)
    
        # Only points that have not already been claimed can be picked as new 
        # seed points.    
        # If the point's label is not 0, continue to the next point.
        if not (labels[P] == 0):
            continue
        
        # Find all of P's neighboring points.
        NeighborPts = regionQuery(X, P, eps)
        
        # If the number is below MinPts, this point is noise. 
        # This is the only condition under which a point is labeled 
        # NOISE--when it's not a valid seed point. A NOISE point may later 
        # be picked up by another cluster as a boundary point (this is the only
        # condition under which a cluster label can change--from NOISE to 
        # something else).
        if len(NeighborPts) < MinPts:
            labels[P] = -1
            ax.scatter(X[P,0],X[P,1],alpha = 0.8,color = 'black',marker = 'x',s = 100)
        # Otherwise, if there are at least MinPts nearby, use this point as the 
        # seed for a new cluster.    
        else: 
            C += 1
            ax.scatter(X[P,0],X[P,1],alpha = 0.8,color = colors[C%len(colors)],marker = '*',s = 100)
            
            col = colors[C%len(colors)] if C != -1 else 'black'
            ax.add_patch(plt.Circle((X[P,0],X[P,1]),eps,fill = False,alpha = 0.6,color = col))
            
            if P_ != -1:
            
                lenx = X[P,0]-X[P_,0]+1e-8
                leny = X[P,1]-X[P_,1]+1e-8

                ax.arrow(X[P_,0],X[P_,1],lenx,leny, length_includes_head=True,head_width=0.1, head_length = 0.1,alpha = 0.5)
            
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png')
            img_buf.detach

            fig_list.append(img_buf)
            
            growCluster(X, labels, P, NeighborPts, C, eps, MinPts, fig, ax, fig_list)
            
            P_ = P
            
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.detach

    fig_list.append(img_buf)
    
    # All data has been clustered!
    return labels,[PIL.Image.open(f) for f in fig_list]


def growCluster(X, labels, P, NeighborPts, C, eps, MinPts, fig, ax, fig_list):
    """
    Grow a new cluster with label `C` from the seed point `P`.
    
    This function searches through the dataset to find all points that belong
    to this new cluster. When this function returns, cluster `C` is complete.
    
    Parameters:
      `D`      - The dataset (a list of vectors)
      `labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `NeighborPts` - All of the neighbors of `P`
      `C`      - The label for this new cluster.  
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
    """

    # Assign the cluster label to the seed point.
    labels[P] = C
    
    # Look at each neighbor of P (neighbors are referred to as Pn). 
    # NeighborPts will be used as a FIFO queue of points to search--that is, it
    # will grow as we discover new branch points for the cluster. The FIFO
    # behavior is accomplished by using a while-loop rather than a for-loop.
    # In NeighborPts, the points are represented by their index in the original
    # dataset.
    i = 0
    Pn_1 = P
    
    while i < len(NeighborPts):  
        
        #print(len(NeighborPts))
        
        # Get the next point from the queue.        
        Pn = NeighborPts[i]
       
        # If Pn was labelled NOISE during the seed search, then we
        # know it's not a branch point (it doesn't have enough neighbors), so
        # make it a leaf point of cluster C and move on.
        if labels[Pn] == -1:
            labels[Pn] = C
            ax.scatter(X[Pn,0],X[Pn,1],alpha = 0.8,color = colors[C%len(colors)])
            
            col = colors[C%len(colors)] if C != -1 else 'black'
            ax.add_patch(plt.Circle((X[Pn,0],X[Pn,1]),eps,fill = False,alpha = 0.6,color = col))

            lenx = X[Pn,0]-X[Pn_1,0]+1e-8
            leny = X[Pn,1]-X[Pn_1,1]+1e-8

            ax.arrow(X[Pn_1,0],X[Pn_1,1],lenx,leny, length_includes_head=True,head_width=0.1, head_length = 0.1,alpha = 0.5)
            
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png')
            img_buf.detach

            fig_list.append(img_buf)
        
        # Otherwise, if Pn isn't already claimed, claim it as part of C.
        elif labels[Pn] == 0:
            # Add Pn to cluster C (Assign cluster label C).
            labels[Pn] = C
            ax.scatter(X[Pn,0],X[Pn,1],alpha = 0.8,color = colors[C%len(colors)])
            
            col = colors[C%len(colors)] if C != -1 else 'black'
            ax.add_patch(plt.Circle((X[Pn,0],X[Pn,1]),eps,fill = False,alpha = 0.6,color = col))

            lenx = X[Pn,0]-X[Pn_1,0]+1e-8
            leny = X[Pn,1]-X[Pn_1,1]+1e-8

            ax.arrow(X[Pn_1,0],X[Pn_1,1],lenx,leny, length_includes_head=True,head_width=0.1, head_length = 0.1,alpha = 0.5)

            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png')
            img_buf.detach

            fig_list.append(img_buf)
            
            # Find all the neighbors of Pn
            PnNeighborPts = regionQuery(X, Pn, eps)
            
            # If Pn has at least MinPts neighbors, it's a branch point!
            # Add all of its neighbors to the FIFO queue to be searched. 
            if len(PnNeighborPts) >= MinPts:
                ax.scatter(X[Pn,0],X[Pn,1],alpha = 0.8,color = colors[C%len(colors)],marker = '*',s = 100)
                
                col = colors[C%len(colors)] if C != -1 else 'black'
                ax.add_patch(plt.Circle((X[Pn,0],X[Pn,1]),eps,fill = False,alpha = 0.6,color = col))
                
                lenx = X[Pn,0]-X[Pn_1,0]+1e-8
                leny = X[Pn,1]-X[Pn_1,1]+1e-8
                
                ax.arrow(X[Pn_1,0],X[Pn_1,1],lenx,leny, length_includes_head=True,head_width=0.1, head_length = 0.1,alpha = 0.5)
                
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format='png')
                img_buf.detach
                
                fig_list.append(img_buf)
                
                NeighborPts = NeighborPts + PnNeighborPts
            # If Pn *doesn't* have enough neighbors, then it's a leaf point.
            # Don't queue up it's neighbors as expansion points.
            #else:
                # Do nothing                
                #NeighborPts = NeighborPts               
        
        # Advance to the next point in the FIFO queue.
        i += 1
        Pn_1 = Pn
    
    # We've finished growing cluster C!


def regionQuery(D, P, eps):
    """
    Find all points in dataset `D` within distance `eps` of point `P`.
    
    This function calculates the distance between a point P and every other 
    point in the dataset, and then returns only those points which are within a
    threshold distance `eps`.
    """
    neighbors = []
    
    # For each point in the dataset...
    for Pn in range(0, len(D)):
        
        # If the distance is below the threshold, add it to the neighbors list.
        if np.linalg.norm(D[P] - D[Pn]) < eps:
            neighbors.append(Pn)
            
    return neighbors