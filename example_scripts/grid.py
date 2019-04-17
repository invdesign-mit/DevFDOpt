import numpy as np
import h5py as hp

class grid:

    def __init__(self,dx,dy,dz,Lx,Ly,Lz):
        self.dx=dx
        self.dy=dy
        self.dz=dz
        x_start=0
        y_start=0
        z_start=0
        x_end=x_start+Lx+dx
        y_end=y_start+Ly+dy
        z_end=z_start+Lz+dz
        self.xraw=np.arange(x_start,x_end,dx)
        self.yraw=np.arange(y_start,y_end,dy)
        self.zraw=np.arange(z_start,z_end,dz)
        self.Nx=self.xraw.size-1
        self.Ny=self.yraw.size-1
        self.Nz=self.zraw.size-1

    def set_bc(self,bcx,bcy,bcz,Lpmlx_neg,Lpmly_neg,Lpmlz_neg,Lpmlx_pos,Lpmly_pos,Lpmlz_pos,kx,ky,kz):
        bcx=bcx
        bcy=bcy
        bcz=bcz
        npmlx_neg=np.round(Lpmlx_neg/self.dx);
        npmly_neg=np.round(Lpmly_neg/self.dy);
        npmlz_neg=np.round(Lpmlz_neg/self.dz);
        npmlx_pos=np.round(Lpmlx_pos/self.dx);
        npmly_pos=np.round(Lpmly_pos/self.dy);
        npmlz_pos=np.round(Lpmlz_pos/self.dz);

        self.bc=np.array((bcx,bcy,bcz))
        self.npml=np.array((npmlx_neg,npmlx_pos,npmly_neg,npmly_pos,npmlz_neg,npmlz_pos)).reshape(3,2)

        Lx=self.Nx*self.dx;
        Ly=self.Ny*self.dy;
        Lz=self.Nz*self.dz;
        eklx=np.exp(1j*kx*Lx)
        ekly=np.exp(1j*ky*Ly)
        eklz=np.exp(1j*kz*Lz)

        self.ekl=np.array((eklx.real,eklx.imag,ekly.real,ekly.imag,eklz.real,eklz.imag)).reshape(3,2)

    def set_dof(self,Mx,My,Nxo,Nyo,nlayers,thickness,Mz,Mzslab):
        self.Mx=Mx
        self.My=My
        self.Nxo=Nxo
        self.Nyo=Nyo
        self.nlayers=nlayers
        self.Mz=np.array(Mz)
        self.Mzslab=Mzslab

        t_tot=np.round(thickness/self.dz)
        t0=np.round(t_tot/nlayers)
        Nzo=np.zeros(nlayers,dtype=int)
        Nzo[0]=int(round((self.Nz-t_tot)/2.0))
        for i in range(1,nlayers):
            Nzo[i]=int(Nzo[i-1]+t0)
        self.Nzo=Nzo
        
    def printh5(self,prefix,freqid,angleid):
        filename=prefix+"_freq"+str(freqid)+"_angle"+str(angleid)+".h5"
        fid=hp.File(filename,"w")
        fid.create_dataset("bc",data=self.bc)
        fid.create_dataset("xraw",data=self.xraw)
        fid.create_dataset("yraw",data=self.yraw)
        fid.create_dataset("zraw",data=self.zraw)
        fid.create_dataset("Npml",data=self.npml)
        fid.create_dataset("e_ikL",data=self.ekl)
        fid.create_dataset("nlayers",data=self.nlayers)
        fid.create_dataset("Mx",data=self.Mx)
        fid.create_dataset("My",data=self.My)
        fid.create_dataset("Mz",data=self.Mz)
        fid.create_dataset("Mzslab",data=self.Mzslab)
        fid.create_dataset("Nxo",data=self.Nxo)
        fid.create_dataset("Nyo",data=self.Nyo)
        fid.create_dataset("Nzo",data=self.Nzo)
        fid.close()

    def printepsBkg(self,name,nsub,nsup):
        eps=np.zeros((self.Nz,self.Ny,self.Nx,3,2))
        eps[0:self.Nzo[nlayers-1]+self.Mz[nlayers-1],:,:,:,0]=nsub*nsub
        eps[self.Nzo[nlayers-1]+self.Mz[nlayers-1]:self.Nz,:,:,:,0]=nsup*nsup
        fid=hp.File(name,"w")
        fid.create_dataset("epsBkg",data=eps)
        fid.close()

    def printepsDiff(self,name,nmed,nsub,nsup):
        eps=np.zeros((self.Nz,self.Ny,self.Nx,3,2))
        for i in range(0,nlayers):
            eps[self.Nzo[i]:self.Nzo[i]+self.Mz[i],:,:,:,0]=nmed*nmed-nsub*nsub
        #eps[self.Nzo[nlayers-1]:self.Nzo[nlayers-1]+self.Mz[nlayers-1],:,:,:,0]=nmed*nmed-nsub*nsub
        fid=hp.File(name,"w")
        fid.create_dataset("epsDiff",data=eps)
        fid.close()
    
