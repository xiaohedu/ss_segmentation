function plot_ss_segmentation_data(filename)
    table_data = dlmread(filename,',',1,0);
    figure,
    plot(1:size(table_data,1), table_data(:,1), 'r', 'LineWidth', 2);
    xlabel('Epochs');
    ylabel('Average Loss');

    figure,
    plot(1:size(table_data,1), table_data(:,5), 'r', 'LineWidth', 2);
    xlabel('Epochs');
    ylabel('Learning rate');

    figure,
    plot(1:size(table_data,1), table_data(:,2), 'r', 'LineWidth', 2);
    hold on
    plot(1:size(table_data,1), table_data(:,3)*100, 'b', 'LineWidth', 2);
    plot(1:size(table_data,1), table_data(:,4)*100, 'g', 'LineWidth', 2);
    xlabel('Epochs');
    ylabel('IoU');
    legend('Average IoU', 'Background IoU', 'Foreground IoU');
end
