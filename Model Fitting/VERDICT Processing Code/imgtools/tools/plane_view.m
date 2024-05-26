function plane_view

fn = pref_uigetfile('plane_view','file') ;

d = datparse(fn) ;

nm = cross(d.ImageOrientationPatient(1:3), d.ImageOrientationPatient(4:6)) 

plot3([0 nm(1)],[0 nm(2)], [0 nm(3)])
hold on 
grid on
axis([-1 1 -1 1 -1 1])
axis equal
axis square
