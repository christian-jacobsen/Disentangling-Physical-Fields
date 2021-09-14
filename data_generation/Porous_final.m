

   corr.name = 'gauss1';
   corr.c0 = 0.1; 
   nx=65;
   ny=65;
   dx=1/(nx-1);
   dy=1/(ny-1);
   x = 0:dx:dx*(nx-1);
   y = 0:dy:dy*(ny-1);
   [X,Y] = meshgrid(x,y); mesh = [X(:) Y(:)]; % 2-D mesh

   corr.sigma = 1.0;    % variance (sill)

   [F,KL,U,L] = randomfield(corr,mesh,'trunc', 50);   % This is where we change the number of KL modes

   F2  =reshape(F,nx,ny);
   Fout=reshape(exp(F),nx,ny);

   [ds, gam] = varcalc(F2,dx);

A = U*L;  
w = F\(A);
w = inv(A'*A)*A'*log(reshape(Fout,nx*ny,1)) % extract the modal coefficients from the field
%{
C = reshape(C, nx, ny);
figure(100)
contourf(X,Y,C)
axis equal
colorbar
%}
figure(1)
contourf(X,Y,Fout)
axis equal
colorbar

figure(2)
hc = histogram(F,20);
pc = histcounts(F,20,'Normalization','pdf');
binCentersc = hc.BinEdges + (hc.BinWidth/2);
plot(binCentersc(1:end-1), pc, 'o')
hold on
vv=-3:0.01:3;
plot(vv,1/sqrt(2*pi*corr.sigma)*exp(-0.5*(vv).^2/corr.sigma),'--')


gg=0:0.01:sqrt(corr.c0);
figure(3)
plot(ds,gam,'o',gg,1-corr.sigma*exp(-gg/corr.c0),'--')
set(gca,'FontSize',16)
xlabel('Distance (km)')
ylabel('Semivariogram')

