import torch
import numpy as np
import hashlib
import matplotlib.pyplot as plt

def plot17j(poses, show_animation=False):
    import matplotlib as mpl
    mpl.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.animation as anim

    from mpl_toolkits.mplot3d import axes3d, Axes3D

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    if not show_animation:
        plot_idx = 1
      
        frames = np.linspace(start=0, stop=poses.shape[0]-1, num=poses.shape[0]).astype(int)
 
        for i in frames:
            ax = fig.add_subplot(1, poses.shape[0], plot_idx, projection='3d')

            x = poses[i,:,0]
            y = poses[i,:,1]
            z = poses[i,:,2]
            

            ax.plot(poses[i,[0,1],0], poses[i,[0,1],1], poses[i,[0,1],2])
            ax.plot(poses[i,[1,2],0], poses[i,[1,2],1], poses[i,[1,2],2])
            ax.plot(poses[i,[3,4],0], poses[i,[3,4],1], poses[i,[3,4],2])
            ax.plot(poses[i,[4,5],0], poses[i,[4,5],1], poses[i,[4,5],2])
            ax.plot(poses[i,[0,6],0], poses[i,[0,6],1], poses[i,[0,6],2])
            ax.plot(poses[i,[3,6],0], poses[i,[3,6],1], poses[i,[3,6],2])
            ax.plot(poses[i,[6,7],0], poses[i,[6,7],1], poses[i,[6,7],2])
            ax.plot(poses[i,[7,8],0], poses[i,[7,8],1], poses[i,[7,8],2])
            ax.plot(poses[i,[8,9],0], poses[i,[8,9],1], poses[i,[8,9],2])
            ax.plot(poses[i,[7,10],0], poses[i,[7,10],1], poses[i,[7,10],2])
            ax.plot(poses[i,[10,11],0], poses[i,[10,11],1], poses[i,[10,11],2])
            ax.plot(poses[i,[11,12],0], poses[i,[11,12],1], poses[i,[11,12],2])
            ax.plot(poses[i,[7,13],0], poses[i,[7,13],1], poses[i,[7,13],2])
            ax.plot(poses[i,[13,14],0], poses[i,[13,14],1], poses[i,[13,14],2])
            ax.plot(poses[i,[14,15],0], poses[i,[14,15],1], poses[i,[14,15],2])

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

            # ax.axis('equal')
            ax.axis('auto')
            ax.axis('on')
            ax.set_title('frame = ' +str(i))
            # ax.view_init(elev=10, azim=90)
            plot_idx += 1

        # this uses QT5Agg backend
        # you can identify the backend using plt.get_backend()
        # delete the following two lines and resize manually if it throws an error
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # plt.grid(True,'r')
        plt.show()

    else:
        def update(i):

            ax.clear()

            pose = poses[i]

            x = pose[0:16]
            y = pose[16:32]
            z = pose[32:48]
            ax.scatter(x, y, z)

            ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])])
            ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])])
            ax.plot(x[([3, 4])], y[([3, 4])], z[([3, 4])])
            ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])])
            ax.plot(x[([0, 6])], y[([0, 6])], z[([0, 6])])
            ax.plot(x[([3, 6])], y[([3, 6])], z[([3, 6])])
            ax.plot(x[([6, 7])], y[([6, 7])], z[([6, 7])])
            ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])])
            ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])])
            ax.plot(x[([7, 10])], y[([7, 10])], z[([7, 10])])
            ax.plot(x[([10, 11])], y[([10, 11])], z[([10, 11])])
            ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])])
            ax.plot(x[([7, 13])], y[([7, 13])], z[([7, 13])])
            ax.plot(x[([13, 14])], y[([13, 14])], z[([13, 14])])
            ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])])

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

            plt.axis('equal')

        a = anim.FuncAnimation(fig, update, frames=poses.shape[0], repeat=False)
        plt.show()
        
    return



def plot15j_PD(poses, show_animation=False,color=None,legend=None):
    import matplotlib as mpl
    mpl.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.animation as anim

    from mpl_toolkits.mplot3d import axes3d, Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if not show_animation:
        plot_idx = 1
      
        frames = np.linspace(start=0, stop=poses.shape[0]-1, num=poses.shape[0]).astype(int)
 
        for i in frames:
            # ax = fig.add_subplot(1, poses.shape[0], plot_idx, projection='3d')

            x = poses[i,:,0]
            y = poses[i,:,1]
            z = poses[i,:,2]
            # ax.scatter3D(x, y, z,color='b')
            if color is None:
                color0='indianred'
                color1='steelblue'
                color2='g'
            else:
                color0,color1,color2=color[i],color[i],color[i]
           

            ax.plot(poses[i,[0,1],0], poses[i,[0,1],1], poses[i,[0,1],2],color0)
            ax.plot(poses[i,[1,2],0], poses[i,[1,2],1], poses[i,[1,2],2],color0)
            ax.plot(poses[i,[2,3],0], poses[i,[2,3],1], poses[i,[2,3],2],color0)
            ax.plot(poses[i,[0,4],0], poses[i,[0,4],1], poses[i,[0,4],2],color1)
            ax.plot(poses[i,[4,5],0], poses[i,[4,5],1], poses[i,[4,5],2],color1)
            ax.plot(poses[i,[5,6],0], poses[i,[5,6],1], poses[i,[5,6],2],color1)
            ax.plot(poses[i,[0,7],0], poses[i,[0,7],1], poses[i,[0,7],2],color2)
            ax.plot(poses[i,[8,7],0], poses[i,[8,7],1], poses[i,[8,7],2],color2)
            ax.plot(poses[i,[7,9],0], poses[i,[7,9],1], poses[i,[7,9],2],color1)
            ax.plot(poses[i,[9,10],0], poses[i,[9,10],1], poses[i,[9,10],2],color1)
            ax.plot(poses[i,[10,11],0], poses[i,[10,11],1], poses[i,[10,11],2],color1)
            ax.plot(poses[i,[7,12],0], poses[i,[7,12],1], poses[i,[7,12],2],color0)
            ax.plot(poses[i,[12,13],0], poses[i,[12,13],1], poses[i,[12,13],2],color0)
            ax.plot(poses[i,[13,14],0], poses[i,[13,14],1], poses[i,[13,14],2],color0)
            # ax.plot(poses[i,[14,15],0], poses[i,[14,15],1], poses[i,[14,15],2],'r')

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')
            # ax.axis('off')
            # ax.axis('equal')
            ax.axis('auto')
            # ax.axis('on')
            # ax.set_title('frame = ' +str(i))
            # ax.view_init(elev=10, azim=90)
            plot_idx += 1

        # this uses QT5Agg backend
        # you can identify the backend using plt.get_backend()
        # delete the following two lines and resize manually if it throws an error
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # plt.grid(True,'r')
        # plt.legend()
        plt.show()

    else:
        def update(i):

            ax.clear()

            pose = poses[i]


            x = poses[i,:,0]
            y = poses[i,:,1]
            z = poses[i,:,2]
            ax.scatter(x, y, z)

            ax.plot(poses[i,[0,1],0], poses[i,[0,1],1], poses[i,[0,1],2],'r')
            ax.plot(poses[i,[1,2],0], poses[i,[1,2],1], poses[i,[1,2],2],'r')
            ax.plot(poses[i,[2,3],0], poses[i,[2,3],1], poses[i,[2,3],2],'r')
            ax.plot(poses[i,[0,4],0], poses[i,[0,4],1], poses[i,[0,4],2],'b')
            ax.plot(poses[i,[4,5],0], poses[i,[4,5],1], poses[i,[4,5],2],'b')
            ax.plot(poses[i,[5,6],0], poses[i,[5,6],1], poses[i,[5,6],2],'b')
            ax.plot(poses[i,[0,7],0], poses[i,[0,7],1], poses[i,[0,7],2],'g')
            ax.plot(poses[i,[8,7],0], poses[i,[8,7],1], poses[i,[8,7],2],'g')
            ax.plot(poses[i,[7,9],0], poses[i,[7,9],1], poses[i,[7,9],2],'b')
            ax.plot(poses[i,[9,10],0], poses[i,[9,10],1], poses[i,[9,10],2],'b')
            ax.plot(poses[i,[10,11],0], poses[i,[10,11],1], poses[i,[10,11],2],'b')
            ax.plot(poses[i,[7,12],0], poses[i,[7,12],1], poses[i,[7,12],2],'r')
            ax.plot(poses[i,[12,13],0], poses[i,[12,13],1], poses[i,[12,13],2],'r')
            ax.plot(poses[i,[13,14],0], poses[i,[13,14],1], poses[i,[13,14],2],'r')

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

            # plt.axis('equal')
     
        ani = anim.FuncAnimation(fig, update, frames=poses.shape[0], repeat=False,blit = False)
        Writer = anim.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('basic_animation.mp4', writer=writer)
        plt.show()
        
    return


def plot17j_2d(poses, show_animation=False):
    import matplotlib as mpl
    mpl.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.animation as anim

    from mpl_toolkits.mplot3d import axes3d, Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if not show_animation:
        plot_idx = 1
# 
        frames = np.linspace(start=0, stop=poses.shape[0], num=poses.shape[0]).astype(int)
        
        for i in range(poses.shape[0]):
            # ax = fig.add_subplot(1, poses.shape[0], plot_idx, projection='3d')

            # pose = poses[i,1:]
            # pose = pose[[2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 12, 11, 10], :]
            # pose=np.reshape(pose,(16*3,))
            
            # x = pose[0:16]
            # y = pose[16:32]
            # z = pose[32:48]

            x = poses[i,:,0]
            y = poses[i,:,1]
        
            ax.scatter(x, y)

            kkkkk=5

            ax.scatter(poses[i,kkkkk,0],poses[i,kkkkk,1])

            linewidth=3

            ax.plot(poses[i,[0,1],0], poses[i,[0,1],1])
            ax.plot(poses[i,[1,2],0], poses[i,[1,2],1])
            ax.plot(poses[i,[3,4],0], poses[i,[3,4],1])
            ax.plot(poses[i,[4,5],0], poses[i,[4,5],1])
            ax.plot(poses[i,[0,6],0], poses[i,[0,6],1])
            ax.plot(poses[i,[3,6],0], poses[i,[3,6],1])
            ax.plot(poses[i,[6,7],0], poses[i,[6,7],1])
            ax.plot(poses[i,[7,8],0], poses[i,[7,8],1])
            ax.plot(poses[i,[8,9],0], poses[i,[8,9],1])
            ax.plot(poses[i,[7,10],0], poses[i,[7,10],1])
            ax.plot(poses[i,[10,11],0], poses[i,[10,11],1])
            ax.plot(poses[i,[11,12],0], poses[i,[11,12],1])
            ax.plot(poses[i,[7,13],0], poses[i,[7,13],1])
            ax.plot(poses[i,[13,14],0], poses[i,[13,14],1])
            ax.plot(poses[i,[14,15],0], poses[i,[14,15],1])

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x.max() - x.min(), y.max() - y.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
        

            for xb, yb in zip(Xb, Yb):
                ax.plot([xb], [yb], 'w')

            # ax.axis('equal')
            ax.axis('auto')
            ax.axis('on')

            ax.set_title('frame = ' + str(i))

            plot_idx += 1

        # this uses QT5Agg backend
        # you can identify the backend using plt.get_backend()
        # delete the following two lines and resize manually if it throws an error
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.grid(True)
        plt.show()

    else:
        def update(i):

            ax.clear()

            pose = poses[i]

            x = pose[0:16]
            y = pose[16:32]
            z = pose[32:48]
            ax.scatter(x, y, z)

            ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])])
            ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])])
            ax.plot(x[([3, 4])], y[([3, 4])], z[([3, 4])])
            ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])])
            ax.plot(x[([0, 6])], y[([0, 6])], z[([0, 6])])
            ax.plot(x[([3, 6])], y[([3, 6])], z[([3, 6])])
            ax.plot(x[([6, 7])], y[([6, 7])], z[([6, 7])])
            ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])])
            ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])])
            ax.plot(x[([7, 10])], y[([7, 10])], z[([7, 10])])
            ax.plot(x[([10, 11])], y[([10, 11])], z[([10, 11])])
            ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])])
            ax.plot(x[([7, 13])], y[([7, 13])], z[([7, 13])])
            ax.plot(x[([13, 14])], y[([13, 14])], z[([13, 14])])
            ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])])

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

            plt.axis('equal')

        ani = anim.FuncAnimation(fig, update, frames=poses.shape[0], repeat=False)
        ani.save('basic_animation.mp4', fps=30)
        plt.show()
    
    return

def plot_15j_3d(poses):
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        plot_idx = 1
        frames = np.linspace(start=0, stop=poses.shape[0]-1, num=2).astype(int)
        for i in range(poses.shape[0]):
                ax = fig.add_subplot(1, poses.shape[0], plot_idx, projection='3d')
                x=poses[i,:,0]
                y=poses[i,:,1]
                z=poses[i,:,2]
                ax.scatter(poses[i,:,0],poses[i,:,1],poses[i,:,2])
                width=1
                ax.plot(poses[i,[0,1],0], poses[i,[0,1],1], poses[i,[0,1],2],linewidth=width,color='red')
                ax.plot(poses[i,[1,2],0], poses[i,[1,2],1], poses[i,[1,2],2],linewidth=width,color='red')
                ax.plot(poses[i,[2,3],0], poses[i,[2,3],1], poses[i,[2,3],2],linewidth=width,color='red')
                ax.plot(poses[i,[0,4],0], poses[i,[0,4],1], poses[i,[0,4],2],linewidth=width,color='blue')
                ax.plot(poses[i,[4,5],0], poses[i,[4,5],1], poses[i,[4,5],2],linewidth=width,color='blue')
                ax.plot(poses[i,[5,6],0], poses[i,[5,6],1], poses[i,[5,6],2],linewidth=width,color='blue')
                ax.plot(poses[i,[0,7],0], poses[i,[0,7],1], poses[i,[0,7],2],linewidth=width,color='green')
                ax.plot(poses[i,[7,8],0], poses[i,[7,8],1], poses[i,[7,8],2],linewidth=width,color='green')
                ax.plot(poses[i,[7,9],0], poses[i,[7,9],1], poses[i,[7,9],2],linewidth=width,color='blue')
                ax.plot(poses[i,[9,10],0], poses[i,[9,10],1], poses[i,[9,10],2],linewidth=width,color='blue')
                ax.plot(poses[i,[10,11],0], poses[i,[10,11],1], poses[i,[10,11],2],linewidth=width,color='blue')
                ax.plot(poses[i,[7,12],0], poses[i,[7,12],1], poses[i,[7,12],2],linewidth=width,color='red')
                ax.plot(poses[i,[12,13],0], poses[i,[12,13],1], poses[i,[12,13],2],linewidth=width,color='red')
                ax.plot(poses[i,[13,14],0], poses[i,[13,14],1], poses[i,[13,14],2],linewidth=width,color='red')
                max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
                # Create cubic bounding box to simulate equal aspect ratio
                max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
                Xb = 0.5 * max_range * np.mgrid[-1:2:4j, -1:2:4j, -1:2:4j][0].flatten() + 0.5 * (x.max() + x.min())
                Yb = 0.5 * max_range * np.mgrid[-1:2:4j, -1:2:4j, -1:2:4j][1].flatten() + 0.5 * (y.max() + y.min())
                Zb = 0.5 * max_range * np.mgrid[-1:2:4j, -1:2:4j, -1:2:4j][2].flatten() + 0.5 * (z.max() + z.min())

                for xb, yb, zb in zip(Xb, Yb, Zb):
                        ax.plot([xb], [yb], [zb], 'w')

                # ax.axis('equal')
                ax.axis('auto')
                # ax.axis('on')
                radius=1.7
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                ax.axes.zaxis.set_ticklabels([])
                ax.grid(True)
                ax.set_title('frame = ' + str(i))
                ax.view_init(elev=15, azim=-70)
                # ax.set_xlim3d([-radius/2, radius/2])
                # ax.set_zlim3d([0, radius])
                # ax.set_ylim3d([-radius/2, radius/2])

                plot_idx += 1
            # this uses QT5Agg backend
        # you can identify the backend using plt.get_backend()
        # delete the following two lines and resize manually if it throws an error
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.grid(True)
        plt.show()

def plot_15j(poses):
        fig = plt.figure()
        # ax = fig.add_subplot(111)
        plot_idx = 1
        frames = np.linspace(start=0, stop=poses.shape[0]-1, num=poses.shape[0]).astype(int)
        for i in range(poses.shape[0]):
                ax = fig.add_subplot(1,poses.shape[0], plot_idx)
                width=1
                x=poses[i,:,0]
                y=poses[i,:,1]
                ax.scatter(poses[i,:,0],poses[i,:,1])
                ax.plot(poses[i,[0,1],0], poses[i,[0,1],1],linewidth=width,color='red')
                ax.plot(poses[i,[1,2],0], poses[i,[1,2],1],linewidth=width,color='red')
                ax.plot(poses[i,[2,3],0], poses[i,[2,3],1],linewidth=width,color='red')
                ax.plot(poses[i,[0,4],0], poses[i,[0,4],1],linewidth=width,color='blue')
                ax.plot(poses[i,[4,5],0], poses[i,[4,5],1],linewidth=width,color='blue')
                ax.plot(poses[i,[5,6],0], poses[i,[5,6],1],linewidth=width,color='blue')
                ax.plot(poses[i,[0,7],0], poses[i,[0,7],1],linewidth=width,color='green')
                ax.plot(poses[i,[7,8],0], poses[i,[7,8],1],linewidth=width,color='green')
                ax.plot(poses[i,[7,9],0], poses[i,[7,9],1],linewidth=width,color='blue')
                ax.plot(poses[i,[9,10],0], poses[i,[9,10],1],linewidth=width,color='blue')
                ax.plot(poses[i,[10,11],0], poses[i,[10,11],1],linewidth=width,color='blue')
                ax.plot(poses[i,[7,12],0], poses[i,[7,12],1],linewidth=width,color='red')
                ax.plot(poses[i,[12,13],0], poses[i,[12,13],1],linewidth=width,color='red')
                ax.plot(poses[i,[13,14],0], poses[i,[13,14],1],linewidth=width,color='red')
                max_range = np.array([x.max() - x.min(), y.max() - y.min()]).max()
                Xb = 0.1 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.1 * (x.max() + x.min())
                Yb = 0.1 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.1 * (y.max() + y.min())
                for xb, yb in zip(Xb, Yb):
                        ax.plot([xb], [yb], 'w')

                # ax.axis('equal')
                ax.axis('auto')
                ax.axis('on')

                ax.set_title('frame = ' + str(i))

                plot_idx += 1
            # this uses QT5Agg backend
        # you can identify the backend using plt.get_backend()
        # delete the following two lines and resize manually if it throws an error
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.grid(True)
        plt.show()

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.inner(v1_u, v2_u), -1.0, 1.0))/3.14*180