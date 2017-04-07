function [newpixels] = Color_DeconvolutionPC(pixels)

% Code updated from ImageJ plug-in by Zhaoxuan.Ma@cshs.org
    
% image size
[height,width,~]=size(pixels);

% GL Haem matrix
MODx(1)= 0.644211; %0.650;
MODy(1)= 0.716556; %0.704;
MODz(1)= 0.266844; %0.286;
% GL Eos matrix
MODx(2)= 0.092789; %0.072;
MODy(2)= 0.954111; %0.990;
MODz(2)= 0.283111; %0.105;

% Zero matrix
MODx(3)= 0.6359544;
MODy(3)= 0.0010;
MODz(3)= 0.7717266;


% start
cosx=zeros(3,1);cosy=zeros(3,1);cosz=zeros(3,1);
len=zeros(3,1);
for i=1:3
    %normalise vector length
    cosx(i)=0;
    cosy(i)=0;
    cosz(i)=0;
    len(i)=sqrt(MODx(i)*MODx(i) + MODy(i)*MODy(i) + MODz(i)*MODz(i));
    if (len(i) ~= 0.0)
        cosx(i)= MODx(i)/len(i);
        cosy(i)= MODy(i)/len(i);
        cosz(i)= MODz(i)/len(i);
    end
end


% translation matrix
if (cosx(2)==0.0) %2nd colour is unspecified
    if (cosy(2)==0.0)
        if (cosz(2)==0.0)
            cosx(2)=cosz(1);
            cosy(2)=cosx(1);
            cosz(2)=cosy(1);
        end
    end
end

if (cosx(3)==0.0)  % 3rd colour is unspecified
    if (cosy(3)==0.0)
        if (cosz(3)==0.0)
            if ((cosx(1)*cosx(1) + cosx(2)*cosx(2))> 1)
                cosx(3)=0.0;
            else
                cosx(3)=sqrt(1.0-(cosx(1)*cosx(1))-(cosx(2)*cosx(2)));
            end

            if ((cosy(1)*cosy(1) + cosy(2)*cosy(2))> 1)
                cosy(3)=0.0;
            else
                cosy(3)=sqrt(1.0-(cosy(1)*cosy(1))-(cosy(2)*cosy(2)));
            end

            if ((cosz(1)*cosz(1) + cosz(2)*cosz(2))> 1)
                cosz(3)=0.0;
            else
                cosz(3)=sqrt(1.0-(cosz(1)*cosz(1))-(cosz(2)*cosz(2)));
            end
        end
    end
end

leng=sqrt(cosx(3)*cosx(3) + cosy(3)*cosy(3) + cosz(3)*cosz(3));

cosx(3)= cosx(3)/leng;
cosy(3)= cosy(3)/leng;
cosz(3)= cosz(3)/leng;

for i=1:3
    if (cosx(i) == 0.0), cosx(i) = 0.001; end
    if (cosy(i) == 0.0), cosy(i) = 0.001; end
    if (cosz(i) == 0.0), cosz(i) = 0.001; end
end


%matrix inversion
A = cosy(2) - cosx(2) * cosy(1) / cosx(1);
V = cosz(2) - cosx(2) * cosz(1) / cosx(1);
C = cosz(3) - cosy(3) * V/A + cosx(3) * (V/A * cosy(1) / cosx(1) - cosz(1) / cosx(1));
q(3) = (-cosx(3) / cosx(1) - cosx(3) / A * cosx(2) / cosx(1) * cosy(1) / cosx(1) + cosy(3) / A * cosx(2) / cosx(1)) / C;
q(2) = -q(3) * V / A - cosx(2) / (cosx(1) * A);
q(1) = 1.0 / cosx(1) - q(2) * cosy(1) / cosx(1) - q(3) * cosz(1) / cosx(1);
q(6) = (-cosy(3) / A + cosx(3) / A * cosy(1) / cosx(1)) / C;
q(5) = -q(6) * V / A + 1.0 / A;
q(4) = -q(5) * cosy(1) / cosx(1) - q(6) * cosz(1) / cosx(1);
q(9) = 1.0 / C;
q(8) = -q(9) * V / A;
q(7) = -q(8) * cosy(1) / cosx(1) - q(9) * cosz(1) / cosx(1);

newpixels=zeros(height,width,3);


% Translate
imagesize=size(pixels);
newpixels=zeros(imagesize);
% log transform the RGB data
Rlog=-((255.0*log((double(pixels(:,:,1))+1)/255.0))/log(255));
Glog=-((255.0*log((double(pixels(:,:,2))+1)/255.0))/log(255));
Blog=-((255.0*log((double(pixels(:,:,3))+1)/255.0))/log(255));

for i=1:3
    % rescale to match original paper
    Rscaled=Rlog*q((i-1)*3+1);
    Gscaled=Glog*q((i-1)*3+2);
    Bscaled=Blog*q(i*3);

    output=exp(-((Rscaled+Gscaled+Bscaled)-255.0)*log(255)/255.0);

    output(output>255)=255;

    newpixels(:,:,i)=floor(output+0.5);
end

newpixels=uint8(newpixels);

    
end
