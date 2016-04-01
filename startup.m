%Requires the FLAM library be in the path
addpath(genpath(pwd));
if exist('rskelf.m','file')~=2
    warning('The FLAM library may not be properly installed')
end
