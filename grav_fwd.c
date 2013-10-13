#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>

#define gamma 1e8*6.67e-11

// Naive sequential kernel
void gr_fun_seq(int Nd, int Ncm, int Nx, int Ny, int Nz, const double* xc,const double* yc, const double*zc, const double* rx, const double* ry, const double* rz,  const int* comp, double* d, const double* m, const double dx, const double dy, const double dz)
{
    int i,j,k,jx,jy,jz,row;
    double dist, dist2, gam, xc0, yc0, zc0, rx0, ry0, rz0;
    gam=gamma*dx*dy*dz;
    
    for (i=0; i<Nd; i++){
        rx0=rx[i];
        ry0=ry[i];
        rz0=rz[i];
        for (jz=0; jz<Nz; jz++){
            zc0=zc[jz];
            for (jy=0; jy<Ny; jy++){
                yc0=yc[jy];
                for (jx=0; jx<Nx; jx++){
                    xc0=xc[jx];
                    dist2=pow(rx0-xc0,2)+pow(ry0-yc0,2)+pow(rz0-zc0,2);
                    dist=sqrt(dist2);
                    for (k=0; k<Ncm; k++){
                        row=i*Ncm+k;
                        j=jx+jy*Nx+jz*Nx*Ny;
                        switch (comp[k]) {
                            
                            case 1: d[row]+=gam/dist2*m[j]; break;
                            case 2: d[row]+=gam*(xc0-rx0)/(dist*dist2)*m[j]; break;
                            case 3: d[row]+=gam*(yc0-ry0)/(dist*dist2)*m[j]; break;
                            case 4: d[row]+=gam*(zc0-rz0)/(dist*dist2)*m[j]; break;
                            case 5: d[row]+=1e4*gam*(2*pow(xc0-rx0,2)-pow(yc0-ry0,2)-pow(zc0-rz0,2))/(dist*dist2*dist2)*m[j]; break;
                            case 6: d[row]+=1e4*gam*(2*pow(yc0-ry0,2)-pow(xc0-rx0,2)-pow(zc0-rz0,2))/(dist*dist2*dist2)*m[j]; break;
                            case 7: d[row]+=1e4*gam*(2*pow(zc0-rz0,2)-pow(xc0-rx0,2)-pow(yc0-ry0,2))/(dist*dist2*dist2)*m[j]; break;
                            case 8: d[row]+=3e4*gam*(xc0-rx0)*(yc0-ry0)/(dist*dist2*dist2)*m[j]; break;
                            case 9: d[row]+=3e4*gam*(xc0-rx0)*(zc0-rz0)/(dist*dist2*dist2)*m[j]; break;
                            case 10:d[row]+=3e4*gam*(zc0-rz0)*(yc0-ry0)/(dist*dist2*dist2)*m[j]; break;
                        }
                    }
                }
            }
        }
    }
}

// Optimized sequential kernel
void gr_fun_seq_opt(int Nd, int Ncm, int Nx, int Ny, int Nz, const double* xc,const double* yc, const double*zc, const double* rx, const double* ry, const double* rz,  const int* comp, double* d, const double* m, const double dx, const double dy, const double dz)
{
    int i,j,k,jx,jy,jz,row;
    double dist, dist2,dist3,dist5, gam, xc0, yc0, zc0, dx0, dy0, dz0;
    gam=gamma*dx*dy*dz;
    
    for (jx=0; jx<Nx; jx++){
        xc0=xc[jx];
        for (jy=0; jy<Ny; jy++){
            yc0=yc[jy];
            for (jz=0; jz<Nz; jz++){
                zc0=zc[jz];
                j=jx+jy*Nx+jz*Nx*Ny;
                for (i=0; i<Nd; i++){
                    dx0=xc0-rx[i];
                    dy0=yc0-ry[i];
                    dz0=zc0-rz[i];
                    dist2=1/(dx0*dx0+dy0*dy0+dz0*dz0);
                    dist=sqrt(dist2); dist3=dist*dist2; dist5=dist3*dist2;
                    for (k=0; k<Ncm; k++){
                        row=i*Ncm+k;
                        switch (comp[k]) {
                            
                            case 1: d[row]+=gam*dist2*m[j]; break;
                            case 2: d[row]+=gam*dx0*(dist3)*m[j]; break;
                            case 3: d[row]+=gam*dy0*(dist3)*m[j]; break;
                            case 4: d[row]+=gam*dz0*(dist3)*m[j]; break;
                            case 5: d[row]+=1e4*gam*(2*dx0*dx0-dy0*dy0-dz0*dz0)*(dist5)*m[j]; break;
                            case 6: d[row]+=1e4*gam*(2*dy0*dy0-dx0*dx0-dz0*dz0)*(dist5)*m[j]; break;
                            case 7: d[row]+=1e4*gam*(2*dz0*dz0-dx0*dx0-dy0*dy0)*(dist5)*m[j]; break;
                            case 8: d[row]+=3e4*gam*dx0*dy0*(dist5)*m[j]; break;
                            case 9: d[row]+=3e4*gam*dx0*dz0*(dist5)*m[j]; break;
                            case 10:d[row]+=3e4*gam*dz0*dy0*(dist5)*m[j]; break;
                        }
                    }
                }
            }
        }
    }
}

// Optimized sequential OMP kernel
void gr_fun_seq_opt_omp(int Nd, int Ncm, int Nx, int Ny, int Nz, const double* xc,const double* yc, const double*zc, const double* rx, const double* ry, const double* rz,  const int* comp, double* d, const double* m, const double dx, const double dy, const double dz)
{
    int i,j,k,jx,jy,jz,row;
    double dist, dist2,dist3,dist5, gam, xc0, yc0, zc0, dx0, dy0, dz0;
    gam=gamma*dx*dy*dz;
    
    for (jx=0; jx<Nx; jx++){
        xc0=xc[jx];
        for (jy=0; jy<Ny; jy++){
            yc0=yc[jy];
            for (jz=0; jz<Nz; jz++){
                zc0=zc[jz];
                j=jx+jy*Nx+jz*Nx*Ny;
                #pragma omp parallel for private(i,dx0,dy0,dz0,dist2,dist,dist3,dist5,k,row)
                for (i=0; i<Nd; i++){
                    dx0=xc0-rx[i];
                    dy0=yc0-ry[i];
                    dz0=zc0-rz[i];
                    dist2=1/(dx0*dx0+dy0*dy0+dz0*dz0);
                    dist=sqrt(dist2); dist3=dist*dist2; dist5=dist3*dist2;
                    for (k=0; k<Ncm; k++){
                        row=i*Ncm+k;
                        switch (comp[k]) {
                            
                            case 1: d[row]+=gam*dist2*m[j]; break;
                            case 2: d[row]+=gam*dx0*(dist3)*m[j]; break;
                            case 3: d[row]+=gam*dy0*(dist3)*m[j]; break;
                            case 4: d[row]+=gam*dz0*(dist3)*m[j]; break;
                            case 5: d[row]+=1e4*gam*(2*dx0*dx0-dy0*dy0-dz0*dz0)*(dist5)*m[j]; break;
                            case 6: d[row]+=1e4*gam*(2*dy0*dy0-dx0*dx0-dz0*dz0)*(dist5)*m[j]; break;
                            case 7: d[row]+=1e4*gam*(2*dz0*dz0-dx0*dx0-dy0*dy0)*(dist5)*m[j]; break;
                            case 8: d[row]+=3e4*gam*dx0*dy0*(dist5)*m[j]; break;
                            case 9: d[row]+=3e4*gam*dx0*dz0*(dist5)*m[j]; break;
                            case 10:d[row]+=3e4*gam*dz0*dy0*(dist5)*m[j]; break;
                        }
                    }
                }
            }
        }
    }
}

// GPU on a CPU kernel with a switch
void gr_fun_gpu_cpu_switch(int Nd, int Ncm, int Nx, int Ny, int Nz, const double* xc,const double* yc, const double*zc, const double* rx, const double* ry, const double* rz,  const int* comp, double* d, const double* m, const double dx, const double dy, const double dz)
{
    int i,j,k,jx,jy,jz,row,ir;
    double dist, dist2,dist3,dist5, gam, xc0, yc0, zc0, dx0, dy0, dz0, dtemp;
    gam=gamma*dx*dy*dz;

    #pragma omp parallel for private(ir,k,i,row,dtemp,jx,xc0,jy,yc0,jz,zc0,j,dx0,dy0,dz0,dist,dist2,dist3,dist5)
    for (ir=0; ir<Nd*Ncm; ir++){
       k = ir/Nd;
       i = ir - k*Nd;
       row=i*Ncm+k;
       dtemp=0;
       for (jx=0; jx<Nx; jx++){
           xc0=xc[jx];
           for (jy=0; jy<Ny; jy++){
               yc0=yc[jy];
               for (jz=0; jz<Nz; jz++){
                   zc0=zc[jz];
                   j=jx+jy*Nx+jz*Nx*Ny;
                    dx0=xc0-rx[i];
                    dy0=yc0-ry[i];
                    dz0=zc0-rz[i];
                    dist2=1/(dx0*dx0+dy0*dy0+dz0*dz0);
                    dist=sqrt(dist2); dist3=dist*dist2; dist5=dist3*dist2;
                        switch (comp[k]) {
                            case 1: dtemp+=gam*dist2*m[j]; break;
                            case 2: dtemp+=gam*dx0*(dist3)*m[j]; break;
                            case 3: dtemp+=gam*dy0*(dist3)*m[j]; break;
                            case 4: dtemp+=gam*dz0*(dist3)*m[j]; break;
                            case 5: dtemp+=1e4*gam*(2*dx0*dx0-dy0*dy0-dz0*dz0)*(dist5)*m[j]; break;
                            case 6: dtemp+=1e4*gam*(2*dy0*dy0-dx0*dx0-dz0*dz0)*(dist5)*m[j]; break;
                            case 7: dtemp+=1e4*gam*(2*dz0*dz0-dx0*dx0-dy0*dy0)*(dist5)*m[j]; break;
                            case 8: dtemp+=3e4*gam*dx0*dy0*(dist5)*m[j]; break;
                            case 9: dtemp+=3e4*gam*dx0*dz0*(dist5)*m[j]; break;
                            case 10:dtemp+=3e4*gam*dz0*dy0*(dist5)*m[j]; break;
                        }
                }
            }
        }
        d[row]=dtemp;
    }
}

// GPU on a CPU kernel
void gr_fun_gpu_cpu(int Nd, int Ncm, int Nx, int Ny, int Nz, const double* xc,const double* yc, const double*zc, const double* rx, const double* ry, const double* rz,  const int* comp, double* d, const double* m, const double dx, const double dy, const double dz)
{
    int i,j,k,jx,jy,jz,row,ir;
    double dist, dist2,dist3,dist5, gam, xc0, yc0, zc0, dx0, dy0, dz0, dtemp;
    gam=gamma*dx*dy*dz;

    #pragma omp parallel for private(ir,k,i,row,dtemp,jx,xc0,jy,yc0,jz,zc0,j,dx0,dy0,dz0,dist,dist2,dist3,dist5)
    for (ir=0; ir<Nd*Ncm; ir++){
       k = ir/Nd;
       i = ir - k*Nd;
       row=i*Ncm+k;
       dtemp=0;
       for (jx=0; jx<Nx; jx++){
           xc0=xc[jx];
           for (jy=0; jy<Ny; jy++){
               yc0=yc[jy];
               for (jz=0; jz<Nz; jz++){
                   zc0=zc[jz];
                   j=jx+jy*Nx+jz*Nx*Ny;
                    dx0=xc0-rx[i];
                    dy0=yc0-ry[i];
                    dz0=zc0-rz[i];
                    dist2=1/(dx0*dx0+dy0*dy0+dz0*dz0);
                    dist=sqrt(dist2); dist3=dist*dist2; dist5=dist3*dist2;
                        if (comp[k]==1) 
                            dtemp+=gam*dist2*m[j]; 
                        else if (comp[k]==2) 
                            dtemp+=gam*dx0*(dist3)*m[j];
                        else if (comp[k]==3) 
                            dtemp+=gam*dy0*(dist3)*m[j];
                        else if (comp[k]==4) 
                            dtemp+=gam*dz0*(dist3)*m[j];
                        else if (comp[k]==5) 
                            dtemp+=1e4*gam*(2*dx0*dx0-dy0*dy0-dz0*dz0)*(dist5)*m[j];
                        else if (comp[k]==6) 
                            dtemp+=1e4*gam*(2*dy0*dy0-dx0*dx0-dz0*dz0)*(dist5)*m[j];
                        else if (comp[k]==7) 
                            dtemp+=1e4*gam*(2*dz0*dz0-dx0*dx0-dy0*dy0)*(dist5)*m[j];
                        else if (comp[k]==8) 
                            dtemp+=3e4*gam*dx0*dy0*(dist5)*m[j];
                        else if (comp[k]==9) 
                            dtemp+=3e4*gam*dx0*dz0*(dist5)*m[j];
                        else if (comp[k]==10) 
                            dtemp+=3e4*gam*dz0*dy0*(dist5)*m[j];
                }
            }
        }
        d[row]=dtemp;
    }
}

#ifdef _OPENACC
// GPU on a CPU kernel
void gr_fun_gpu(int Nd, int Ncm, int Nx, int Ny, int Nz, const double* x,const double* y, const double*z, const double* rx, const double* ry, const double* rz,  const int* comp, double* d, const double* m, const double dx, const double dy, const double dz)
{
    int i,j,k,jx,jy,jz,row,ir;
    double dist, dist2,dist3,dist5, gam, xc0, yc0, zc0, dx0, dy0, dz0, dtemp;
    gam=gamma*dx*dy*dz;

    #pragma acc kernels loop independent present(x[Nx],y[Ny],z[Nz],rx[Nd],ry[Nd],rz[Nd], comp[Ncm]) copyin(m[Nx*Ny*Nz]) copyout(d[Nd*Ncm])
    for (ir=0; ir<Nd*Ncm; ir++){
       k = ir/Nd;
       i = ir - k*Nd;
       row=i*Ncm+k;
       dtemp=0.;
       #pragma acc loop seq
       for (jx=0; jx<Nx; jx++){
           xc0=x[jx];
           #pragma acc loop seq
           for (jy=0; jy<Ny; jy++){
               yc0=y[jy];
               #pragma acc loop seq
               for (jz=0; jz<Nz; jz++){
                   zc0=z[jz];
                   j=jx+jy*Nx+jz*Nx*Ny;
                    dx0=xc0-rx[i];
                    dy0=yc0-ry[i];
                    dz0=zc0-rz[i];
                    dist2=1/(dx0*dx0+dy0*dy0+dz0*dz0);
                    dist=sqrt(dist2); dist3=dist*dist2; dist5=dist3*dist2;
                        if (comp[k]==1) 
                            dtemp+=gam*dist2*m[j]; 
                        else if (comp[k]==2) 
                            dtemp+=gam*dx0*(dist3)*m[j];
                        else if (comp[k]==3) 
                            dtemp+=gam*dy0*(dist3)*m[j];
                        else if (comp[k]==4) 
                            dtemp+=gam*dz0*(dist3)*m[j];
                        else if (comp[k]==5) 
                            dtemp+=1e4*gam*(2*dx0*dx0-dy0*dy0-dz0*dz0)*(dist5)*m[j];
                        else if (comp[k]==6) 
                            dtemp+=1e4*gam*(2*dy0*dy0-dx0*dx0-dz0*dz0)*(dist5)*m[j];
                        else if (comp[k]==7) 
                            dtemp+=1e4*gam*(2*dz0*dz0-dx0*dx0-dy0*dy0)*(dist5)*m[j];
                        else if (comp[k]==8) 
                            dtemp+=3e4*gam*dx0*dy0*(dist5)*m[j];
                        else if (comp[k]==9) 
                            dtemp+=3e4*gam*dx0*dz0*(dist5)*m[j];
                        else if (comp[k]==10) 
                            dtemp+=3e4*gam*dz0*dy0*(dist5)*m[j];
                }
            }
        }
        d[row]=dtemp;
    }
}
#endif

void gr_fun_vec(int Nd, int Ncm, int Nx, int Ny, int Nz, const double* xc,const double* yc, const double*zc, const double* rx, const double* ry, const double* rz, const int* comp2, double* d, const double* m, const double dx, const double dy, const double dz)
{
    int i,j,k,jx,jy,jz;
    double dist, dist2, gam, xc0, yc0, zc0, rx0, ry0, rz0, m0;
    double d_tmp[10];
    gam=gamma*dx*dy*dz;
    
    for (i=0; i<Nd; i++) {
        
        for (k=0; k<10; k++) {
            d_tmp[k]=0;
        }

        for (jz=0; jz<Nz; jz++) {
            for (jy=0; jy<Ny; jy++) {
#pragma ivdep
                for (jx=0; jx<Nx; jx++) {
                    j=jx+jy*Nx+jz*Nx*Ny;
                    dist2=pow(rx[i]-xc[jx],2)+pow(ry[i]-yc[jy],2)+pow(rz[i]-zc[jz],2);
                    dist=sqrt(dist2);
                    d_tmp[0]+=gam/dist2*m[j];
                    d_tmp[1]+=gam*(xc[jx]-rx[i])/(dist*dist2)*m[j];
                    d_tmp[2]+=gam*(yc[jy]-ry[i])/(dist*dist2)*m[j];
                    d_tmp[3]+=gam*(zc[jz]-rz[i])/(dist*dist2)*m[j];
                    d_tmp[4]+=1e4*gam*(2*pow(xc[jx]-rx[i],2)-pow(yc[jy]-ry[i],2)-pow(zc[jz]-rz[i],2))/(dist*dist2*dist2)*m[j];
                    d_tmp[5]+=1e4*gam*(2*pow(yc[jy]-ry[i],2)-pow(xc[jx]-rx[i],2)-pow(zc[jz]-rz[i],2))/(dist*dist2*dist2)*m[j];
                    d_tmp[6]+=1e4*gam*(2*pow(zc[jz]-rz[i],2)-pow(xc[jx]-rx[i],2)-pow(yc[jy]-ry[i],2))/(dist*dist2*dist2)*m[j];
                    d_tmp[7]+=3e4*gam*(xc[jx]-rx[i])*(yc[jy]-ry[i])/(dist*dist2*dist2)*m[j];
                    d_tmp[8]+=3e4*gam*(xc[jx]-rx[i])*(zc[jz]-rz[i])/(dist*dist2*dist2)*m[j];
                    d_tmp[9]+=3e4*gam*(zc[jz]-rz[i])*(yc[jy]-ry[i])/(dist*dist2*dist2)*m[j];
                }
            }
        }
        
        for (k=0; k<10; k++) {
            d[i*10+k]=d_tmp[k];
        }
    }    
}

void gr_fun_vec_omp(int Nd, int Ncm, int Nx, int Ny, int Nz, const double* xc,const double* yc, const double*zc, const double* rx, const double* ry, const double* rz, const int* comp2, double* d, const double* m, const double dx, const double dy, const double dz)
{
    int i,j,k,jx,jy,jz;
    double dist, dist2, gam;
    double d_tmp[10];
    gam=gamma*dx*dy*dz;
    
#pragma omp parallel for private(j,dist2,dist,d_tmp) shared(rx,ry,rz,xc,yc,zc,d,m) 
    for (i=0; i<Nd; i++) {
        for (k=0; k<10; k++) {
            d_tmp[k]=0;
        }
        for (jz=0; jz<Nz; jz++) {
            for (jy=0; jy<Ny; jy++) {
                #pragma ivdep
                for (jx=0; jx<Nx; jx++) {
                    j=jx+jy*Nx+jz*Nx*Ny;
                    dist2=pow(rx[i]-xc[jx],2)+pow(ry[i]-yc[jy],2)+pow(rz[i]-zc[jz],2);
                    dist=sqrt(dist2);
                    d_tmp[0]+=gam/dist2*m[j];
                    d_tmp[1]+=gam*(xc[jx]-rx[i])/(dist*dist2)*m[j];
                    d_tmp[2]+=gam*(yc[jy]-ry[i])/(dist*dist2)*m[j];
                    d_tmp[3]+=gam*(zc[jz]-rz[i])/(dist*dist2)*m[j];
                    d_tmp[4]+=1e4*gam*(2*pow(xc[jx]-rx[i],2)-pow(yc[jy]-ry[i],2)-pow(zc[jz]-rz[i],2))/(dist*dist2*dist2)*m[j];
                    d_tmp[5]+=1e4*gam*(2*pow(yc[jy]-ry[i],2)-pow(xc[jx]-rx[i],2)-pow(zc[jz]-rz[i],2))/(dist*dist2*dist2)*m[j];
                    d_tmp[6]+=1e4*gam*(2*pow(zc[jz]-rz[i],2)-pow(xc[jx]-rx[i],2)-pow(yc[jy]-ry[i],2))/(dist*dist2*dist2)*m[j];
                    d_tmp[7]+=3e4*gam*(xc[jx]-rx[i])*(yc[jy]-ry[i])/(dist*dist2*dist2)*m[j];
                    d_tmp[8]+=3e4*gam*(xc[jx]-rx[i])*(zc[jz]-rz[i])/(dist*dist2*dist2)*m[j];
                    d_tmp[9]+=3e4*gam*(zc[jz]-rz[i])*(yc[jy]-ry[i])/(dist*dist2*dist2)*m[j];
                }
            }
        }
        for (k=0; k<10; k++) {
            d[i*10+k]=d_tmp[k];
        }
    }
}

int main(int argc, char** argv)
{
    int Nr, Ncm, Nd, Nx, Ny, Nz, Nc;
    double anomBounds[6]={-250, 250, -250, 250, 0, 200};  // Anomalous domain size
    double xySize[2]={25/1, 25/1};                        // X, Y cell size, change to adjust problem size 
    double dz=25/1;                                       // Z cell size, change to adjust problem size
    double anoms=1;
    int i,j,count;
    FILE *fp;

//    clock_t start, end;
    double start, end;
    struct timeval _ttime;
    struct timezone _tzone;

    
    Nx=(anomBounds[1]-anomBounds[0])/xySize[0];
    Ny=(anomBounds[3]-anomBounds[2])/xySize[1];
    Nz=(anomBounds[5]-anomBounds[4])/dz;
    Nc=Nx*Ny*Nz;
       
    // Discretize 3D inversion domain
    double *x,*y,*z;
    x=(double*) malloc(sizeof(double)*Nx);
    y=(double*) malloc(sizeof(double)*Ny);
    z=(double*) malloc(sizeof(double)*Nz);
    x[0]=anomBounds[0]+xySize[0]/2;
    y[0]=anomBounds[2]+xySize[1]/2;
    z[0]=anomBounds[4]+dz/2;
    for (i=1; i<Nx; i++)
        x[i]=x[i-1]+xySize[0];
    for (i=1; i<Ny; i++)
        y[i]=y[i-1]+xySize[1];
    for (i=1; i<Nz; i++)
        z[i]=z[i-1]+dz;

    // Assign model properties
    double* m;
    m=(double*)malloc(sizeof(double)*Nc);
    for (i=0; i<Nc; i++)
        m[i]=anoms; 
    
    // Set receiver positions, change to adjust problem size
    Nr=21*4;
    double rxx[Nr], ryy[Nr];
    rxx[0]=-500; ryy[0]=-500;
    for (i=1; i<Nr; i++){
        rxx[i]=rxx[i-1]+50.0/4;
        ryy[i]=ryy[i-1]+50.0/4;
    }
    
    Nd=Nr*Nr;
    double *rx,*ry,*rz;
    rx=(double*)malloc(sizeof(double)*Nd);
    ry=(double*)malloc(sizeof(double)*Nd);
    rz=(double*)malloc(sizeof(double)*Nd);
    for (j=0; j<Nr; j++){
        for (i=0; i<Nr; i++){
            rx[i+j*Nr]=rxx[i];
            ry[i+j*Nr]=ryy[j];
            rz[i+j*Nr]=-50;
        }
    }
    
    // Set observed data components
    Ncm=10;
    int comp[10]={1,2,3,4,5,6,7,8,9,10};
    double *d=(double*)malloc(sizeof(double)*Nd*Ncm);
    for(i=0;i<Nd*Ncm;i++){
      d[i]=0;
    }
    
    // Timing for the sequenial kernel
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
//    start = clock();
    gr_fun_seq(Nd, Ncm, Nx, Ny, Nz, x, y, z, rx, ry, rz, comp, d, m, xySize[0], xySize[1], dz);
    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
//    end = clock();
    printf("CPU kernel seq elapsed time = %lf sec\n", (double)(end - start) / (CLOCKS_PER_SEC/1000));
    
    // Write result by kernel 1
    if((fp=fopen("outData_seq.dat","wt"))!=NULL){
        for (i=0; i<Nd*Ncm; i++){
            fprintf(fp, "  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e\n", rx[i/Ncm], ry[i/Ncm], rz[i/Ncm], (double)comp[i%Ncm], d[i]);
        }
    }
    fclose(fp);
    
    int comp2[10];
    for (i=0; i<10; i++) {
        comp2[i]=0;
    }
    for (i=0; i<Ncm; i++) {
        comp2[comp[i]-1]=1;
    }
    double *d2=(double *) malloc(sizeof(double)*Nd*10);
    for(i=0;i<Nd*10;i++){
        d2[i]=0;
    }
    
    // Timing for the vectorized kernel
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
//    start = clock();
    gr_fun_vec(Nd, Ncm, Nx, Ny, Nz, x, y, z, rx, ry, rz, comp2, d2, m, xySize[0], xySize[1], dz);
//    end = clock();
    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    printf("CPU kernel vec elapsed time = %lf sec\n", (double)(end - start) / (CLOCKS_PER_SEC/1000));
    
    // Write result by kernel 5    
    if((fp=fopen("outData_vec.dat","wt"))!=NULL){
        for (i=0; i<Nd*10; i++){
            if (comp2[i%10]>0) {
//                fprintf(fp, "  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e\n", rx[i/10], ry[i/10], rz[i/10], (double)(i%10+1), d2[i/10+i%10*Nd]);
                fprintf(fp, "  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e\n", rx[i/10], ry[i/10], rz[i/10], (double)(i%10+1), d2[i]);
                
            }
        }
    }
    fclose(fp);
    
    for(i=0;i<Nd*10;i++){
        d2[i]=0;
    }
    
    // Timing for the vectorized OMP kernel
//    start = clock();
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    gr_fun_vec_omp(Nd, Ncm, Nx, Ny, Nz, x, y, z, rx, ry, rz, comp2, d2, m, xySize[0], xySize[1], dz);
//    end = clock();
    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    printf("CPU kernel vec omp elapsed time = %lf sec\n", (double)(end - start) / (CLOCKS_PER_SEC/1000));
    
    // Write result by kernel 9
    if((fp=fopen("outData_vec_omp.dat","wt"))!=NULL){
        for (i=0; i<Nd*10; i++){
            if (comp2[i%10]>0) {
                fprintf(fp, "  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e\n", rx[i/10], ry[i/10], rz[i/10], (double)(i%10+1), d2[i]);
                
            }
        }
    }
    fclose(fp);
    
    for(i=0;i<Nd*Ncm;i++){
      d[i]=0;
    }
    
    // Timing for the sequenial optimized kernel
//    start = clock();
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    gr_fun_seq_opt(Nd, Ncm, Nx, Ny, Nz, x, y, z, rx, ry, rz, comp, d, m, xySize[0], xySize[1], dz);
//    end = clock();
    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    printf("CPU kernel seq opt elapsed time = %lf sec\n", (double)(end - start) / (CLOCKS_PER_SEC/1000));
    
    // Write result by kernel 1
    if((fp=fopen("outData_seq_opt.dat","wt"))!=NULL){
        for (i=0; i<Nd*Ncm; i++){
            fprintf(fp, "  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e\n", rx[i/Ncm], ry[i/Ncm], rz[i/Ncm], (double)comp[i%Ncm], d[i]);
        }
    }
    fclose(fp);
    
    for(i=0;i<Nd*Ncm;i++){
      d[i]=0;
    }
    
    // Timing for the sequenial optimized OMP kernel
//    start = clock();
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    gr_fun_seq_opt_omp(Nd, Ncm, Nx, Ny, Nz, x, y, z, rx, ry, rz, comp, d, m, xySize[0], xySize[1], dz);
//    end = clock();
    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    printf("CPU kernel seq opt OMP elapsed time = %lf sec\n", (double)(end - start) / (CLOCKS_PER_SEC/1000));
    
    // Write result by kernel 1
    if((fp=fopen("outData_seq_opt_omp.dat","wt"))!=NULL){
        for (i=0; i<Nd*Ncm; i++){
            fprintf(fp, "  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e\n", rx[i/Ncm], ry[i/Ncm], rz[i/Ncm], (double)comp[i%Ncm], d[i]);
        }
    }
    fclose(fp);
    
    for(i=0;i<Nd*Ncm;i++){
      d[i]=0;
    }
    // Timing for the GPU kernel on CPU
//    start = clock();
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    gr_fun_gpu_cpu_switch(Nd, Ncm, Nx, Ny, Nz, x, y, z, rx, ry, rz, comp, d, m, xySize[0], xySize[1], dz);
//    end = clock();
    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    printf("CPU kernel gpu on cpu switch elapsed time = %lf sec\n", (double)(end - start) / (CLOCKS_PER_SEC/1000));
    
    // Write result by kernel 1
    if((fp=fopen("outData_gpu_cpu_switch.dat","wt"))!=NULL){
        for (i=0; i<Nd*Ncm; i++){
            fprintf(fp, "  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e\n", rx[i/Ncm], ry[i/Ncm], rz[i/Ncm], (double)comp[i%Ncm], d[i]);
        }
    }
    fclose(fp);
    
    for(i=0;i<Nd*Ncm;i++){
      d[i]=0;
    }
    // Timing for the GPU kernel on CPU
//    start = clock();
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    gr_fun_gpu_cpu(Nd, Ncm, Nx, Ny, Nz, x, y, z, rx, ry, rz, comp, d, m, xySize[0], xySize[1], dz);
//    end = clock();
    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    printf("CPU kernel gpu on cpu elapsed time = %lf sec\n", (double)(end - start) / (CLOCKS_PER_SEC/1000));
    
    // Write result by kernel 1
    if((fp=fopen("outData_gpu_cpu.dat","wt"))!=NULL){
        for (i=0; i<Nd*Ncm; i++){
            fprintf(fp, "  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e\n", rx[i/Ncm], ry[i/Ncm], rz[i/Ncm], (double)comp[i%Ncm], d[i]);
        }
    }
    fclose(fp);
    
#ifdef _OPENACC
    for(i=0;i<Nd*Ncm;i++){
      d[i]=0;
    }
    
//  OpenACC data region, time the copyin
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    #pragma acc data copyin(x[Nx],y[Ny],z[Nz],rx[Nd],ry[Nd],rz[Nd],comp[Ncm])
    {
    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    printf("GPU data copyin time = %lf sec\n", (double)(end - start) / (CLOCKS_PER_SEC/1000));

    // Timing for the GPU kernel on CPU
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    gr_fun_gpu(Nd, Ncm, Nx, Ny, Nz, x, y, z, rx, ry, rz, comp, d, m, xySize[0], xySize[1], dz);
    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    printf("GPU kernel elapsed time = %lf sec\n", (double)(end - start) / (CLOCKS_PER_SEC/1000));
    }
    // Write result by kernel 1
    if((fp=fopen("outData_gpu.dat","wt"))!=NULL){
        for (i=0; i<Nd*Ncm; i++){
            fprintf(fp, "  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e\n", rx[i/Ncm], ry[i/Ncm], rz[i/Ncm], (double)comp[i%Ncm], d[i]);
        }
    }
    fclose(fp);
#endif
    
    free(x); free(y); free(z);
    free(rx); free(ry); free(rz);
    free(m); free(d); free(d2);

    return 1;
}
