clear; clc; close all;

figure('name', 'Standard Map');
sz = 3;

pic_num = 1;
for K = 0.1 : 0.1 : 5
    for initial_position = 0 : 0.2 : 2 * pi
        for initial_momentum = 0 : 0.2 : 2 * pi
            position(1) = initial_position; momentum(1) = initial_momentum;
            for n = 1 : 1000          
                momentum(n + 1) = momentum(n) + K * sin(position(n));
                momentum(n + 1) = mod(momentum(n + 1), 2 * pi);
                position(n + 1) = position(n) + momentum(n + 1);
                position(n + 1) = mod(position(n + 1), 2 * pi);
            end
            scatter(position, momentum, sz, 'filled');
            hold on;
        end
    end
    title(['K = ', num2str(K)], 'FontSize', 20, 'FontName', 'Times New Roman');
    axis([0, 2 * pi, 0, 2 * pi]);
    xlabel('\theta', 'FontSize', 20, 'FontName', 'Times New Roman');
    ylabel('p', 'FontSize', 20, 'FontName', 'Times New Roman'); 
    
    F = getframe(gcf);
    I = frame2im(F);
    
    [I, map] = rgb2ind(I, 256);
    
    if pic_num == 1
        imwrite(I, map, 'StandardMap.gif', 'Loopcount', inf, 'DelayTime', 0.2);
    else
        imwrite(I, map, 'StandardMap.gif', 'WriteMode', 'append', 'Delaytime', 0.2);
    end
    pic_num = pic_num + 1;
end
    
    
    
    