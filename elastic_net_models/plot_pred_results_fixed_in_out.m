function plot_pred_results_fixed_in_out(training_error, true_pred_train, test_error_in, true_pred_in,test_error_out, true_pred_out, title_)
    
    font_size = 15;
    font_name = 'Avenir';
    markersize = 100;
    % Plot model fitting results
    % True vs predicted
    figure("Position",[0, 0, 400, 400])
    hold on
    box on
    p0 = scatter(true_pred_train(:,1),true_pred_train(:,2),markersize,'filled','o','MarkerFaceColor','#2E86C1','MarkerEdgeColor','k','DisplayName','Training');
    p1 = scatter(true_pred_in(:,1),true_pred_in(:,2),markersize,'filled','d','MarkerFaceColor','#F39C12','MarkerEdgeColor','k','DisplayName','High DoD');
    p2 = scatter(true_pred_out(:,1),true_pred_out(:,2),markersize,'filled','square','MarkerFaceColor','#2ECC71','MarkerEdgeColor','k','DisplayName','Low DoD');
    
    plot(0:70,0:70,'k')
    xlabel('True lifetime [weeks]')
    ylabel('Predicted lifetime [weeks]')
    set(gca, 'FontSize', font_size, 'FontName',font_name,'XScale','log','YScale','log')
    title(title_)
    legend([p0,p1,p2],'Location','southeast')
    hold off
    xlim([3,70])
    ylim([3,70])
    xticks([10,20,40,60])
    yticks([10,20,40,60])

    T = table([training_error(1);test_error_in(1);test_error_out(1)],[training_error(2);test_error_in(2);test_error_out(2)],[training_error(3);test_error_in(3);test_error_out(3)],'VariableNames',{'MAE','MAPE','RMSE'},'RowName',{'Training','In-distribution','Out-of-distribution'}); 
    disp(T)
    
end