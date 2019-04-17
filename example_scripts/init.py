import os
execfile("grid.py")

ncells=40
np_per_cell=1
total_np=ncells*np_per_cell
print "total # of processes must be " + str(total_np)

freqs=np.array([1.0])
angles=np.array([0,10,30,20,20])
nmed=np.array([1.00])
nsub=np.array([1.53])
nsup=np.array([1.00])
dof_file="mid.txt"

Job=1

opt_ifreq=np.array([0])
opt_jangle=np.array([4])
print_dof=10

print_ifreq=0
print_jangle=0
print_Efields=0

test_ifreq=0
test_jangle=0

dx=0.02
dy=0.02
dz=0.02
Lxraw=5.0
Lyraw=dy
Lzraw=6.0
bcx=2
bcy=2
bcz=2
pmlx=[0,0]
pmly=[0,0]
pmlz=[0.5,0.5]
ky=0
kz=0

gtmp=grid(dx,dy,dz,Lxraw,Lyraw,Lzraw)
Mx=gtmp.Nx
My=gtmp.Ny
Nxo=0
Nyo=0
nlayers=3
thickness=4.2
Mz=[35,35,35] #measured in number of pixels
Mzslab=1

filter_choice=0
filter_radius= 1  #measured in pixels
filter_threshold=0.5
filter_steepness=0

prefix="2dcell"
foldername="domainfiles"
config_name="config"

focal_length = 100
ff_x= np.multiply(focal_length,np.tan(np.multiply(-angles,np.pi/180)))
ff_z= focal_length * np.ones(freqs.size*angles.size)
gap_src_pml=0.2
gap_pml_ref=0.2

alg=24
localalg=24
maxeval=1001
maxtime=100000

############################
def writeflag(pre,x):
    strx=pre+" "+str(x[0])
    for y in x[1::]:
        strx+=","+str(y)
    strx+="\n"
    return strx

zjref=pmlz[0]+gap_src_pml
zpref=gtmp.Nz*gtmp.dz-(pmlz[1]+gap_pml_ref)
gtmp.set_dof(Mx,My,Nxo,Nyo,nlayers,thickness,Mz,Mzslab)

dof=0.0*np.ones((ncells,My,Mx,nlayers))
np.savetxt("vac.txt",dof.flatten())
dof=0.5*np.ones((ncells,My,Mx,nlayers))
np.savetxt("mid.txt",dof.flatten())
dof=1.0*np.ones((ncells,My,Mx,nlayers))
np.savetxt("full.txt",dof.flatten())
dof=np.random.rand(ncells,My,Mx,nlayers)
np.savetxt("rand.txt",dof.flatten())

fid=open(config_name,"w")
fid.write("-numcells "+str(ncells)+"\n")
fid.write("-inputfile_name \""+foldername+"/"+prefix+"\"\n")
fid.write(writeflag("-freqs",freqs))
fid.write(writeflag("-angles",angles))
fid.write(writeflag("-nsub",nsub))
fid.write(writeflag("-ffx",ff_x))
fid.write(writeflag("-ffz",ff_z))
fid.write("-hx "+str(gtmp.dx)+"\n")
fid.write("-hz "+str(gtmp.dz)+"\n")
fid.write("-zjref "+str(zjref)+"\n")
fid.write("-zpref "+str(zpref)+"\n")
fid.write("-init_dof_name \""+dof_file+"\"\n")
fid.write("-Job "+str(Job)+"\n")
fid.write("-print_ifreq "+str(print_ifreq)+"\n")
fid.write("-print_jangle "+str(print_jangle)+"\n")
fid.write("-print_Efields "+str(print_Efields)+"\n")
fid.write("-test_ifreq "+str(test_ifreq)+"\n")
fid.write("-test_jangle "+str(test_jangle)+"\n")
fid.write(writeflag("-opt_ifreq",opt_ifreq))
fid.write(writeflag("-opt_jangle",opt_jangle))
fid.write("-print_dof "+str(print_dof)+"\n")
fid.write("-alg "+str(alg)+"\n")
fid.write("-localalg "+str(localalg)+"\n")
fid.write("-maxeval "+str(maxeval)+"\n")
fid.write("-maxtime "+str(maxtime)+"\n")
fid.write("-filter_choice "+str(filter_choice)+"\n")
fid.write("-filter_radius_in_pixels "+str(filter_radius)+"\n")
fid.write("-filter_threshold "+str(filter_threshold)+"\n")
fid.write("-filter_steepness "+str(filter_steepness)+"\n")
fid.close()

for freqid in range(0,freqs.size):
    name=prefix+"_freq"+str(freqid)
    gtmp.printepsBkg(name+"_epsBkg.h5",nsub[freqid],nsup[freqid])
    gtmp.printepsDiff(name+"_epsDiff.h5",nmed[freqid],nsub[freqid],nsup[freqid])
    for thetaid in range(0,angles.size):
        f=freqs[freqid]
        theta=angles[thetaid]
        kx = 2*np.pi*f*nsub[freqid]*np.sin(theta*np.pi/180)
        gtmp.set_bc(bcx,bcy,bcz,pmlx[0],pmly[0],pmlz[0],pmlx[1],pmly[1],pmlz[1],kx,ky,kz)
        gtmp.printh5(prefix,freqid,thetaid)

os.system("rm -rf "+foldername)
os.system("mkdir "+foldername)
os.system("mv "+prefix+"*.h5 "+foldername+"/")

cmd="sed -i \"/numcells=/c\\numcells="+str(ncells)+"\" join.py"
os.system(cmd)
cmd="sed -i \"/print_Efield=/c\\print_Efield="+str(print_Efields)+"\" join.py"
os.system(cmd)

#np_per_node=4
#nodes=total_np/np_per_node
#stime="12:00:00"
#cmd="sed -i \"/#SBATCH -N /c\#SBATCH -N "+str(nodes)+"\" run.sh"
#os.system(cmd)
#cmd="sed -i \"/#SBATCH --ntasks/c\#SBATCH --ntasks-per-node="+str(np_per_node)+"\" run.sh"
#os.system(cmd)
#cmd="sed -i \"/#SBATCH --time/c\#SBATCH --time="+stime+"\" run.sh"
#os.system(cmd)

