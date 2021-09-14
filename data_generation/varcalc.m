function[ds, gam] = varcalc(F2,dx)
   nx=size(F2,1)
   ny=size(F2,2)

   var1=0;
   for i=1:ny
   var1=var1+0.5*sum(diff(F2(:,i)).^2);
   end
   for i=1:nx
   var1=var1+0.5*sum(diff(F2(i,:)).^2);
   end
   var1=var1/((nx-1)*ny+(ny-1)*nx)

   var2=0;
   for i=1:ny
   var2=var2+0.5*sum(diff(F2(1:2:nx,i)).^2);
   end
   for i=1:nx
   var2=var2+0.5*sum(diff(F2(i,1:2:ny)).^2);
   end
   var2=var2/((nx-1)*ny/2+(ny-1)/2*nx)

   var4=0;
   for i=1:ny
   var4=var4+0.5*sum(diff(F2(1:4:nx,i)).^2);
   end
   for i=1:nx
   var4=var4+0.5*sum(diff(F2(i,1:4:ny)).^2);
   end
   var4=var4/((nx-1)*ny/4+(ny-1)/4*nx)

   var8=0;
   for i=1:ny
   var8=var8+0.5*sum(diff(F2(1:8:nx,i)).^2);
   end
   for i=1:nx
   var8=var8+0.5*sum(diff(F2(i,1:8:ny)).^2);
   end
   var8=var8/((nx-1)*ny/8+(ny-1)/8*nx)

   var16=0;
   for i=1:ny
   var16=var16+0.5*sum(diff(F2(1:16:nx,i)).^2);
   end
   for i=1:nx
   var16=var16+0.5*sum(diff(F2(i,1:16:ny)).^2);
   end
   var16=var16/((nx-1)*ny/16+(ny-1)/16*nx)


ds = [dx 2*dx 4*dx 8*dx 16*dx] ;
gam = [var1 var2 var4 var8 var16] ;

