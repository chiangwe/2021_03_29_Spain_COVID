function [covid_tr, covid_te, n_cty, n_day_tr, n_day_te] = HawkeKconst_input(type_case, d_pred_start, delta)
warning('OFF', 'MATLAB:table:ModifiedAndSavedVarnames')


% Get last day as pred_end day
last_day = readtable( './Input/Infect.csv','ReadVariableNames',true, 'Range', '1:2').Properties.VariableNames(end);
last_day = last_day{:};
last_day = strrep(last_day,'_','-');

mobi_out = ['./imputation_temp_dir/mobi_' type_case  '_pd_start_' d_pred_start '_last_day_' last_day '.csv']

%% Save out

if strcmpi(type_case, 'confirm')
    NYT = readtable(['./Input/Infect.csv'],'ReadVariableNames',true);
else
    NYT = readtable(['./Input/Death.csv'],'ReadVariableNames',true);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
NYT_Date_list = NYT.Properties.VariableNames(7:end);
NYT_Key_list = table2cell(NYT(:,1:6));
%
NYT_val = table2array( NYT(:,7:end));
NYT_val(isnan(NYT_val)) = 0;
%
%% Data Pre-Processing
covid = NYT_val;

% Calculate the difference from accumulative sum of cases
% The first day is padded with zero (i.e., assusme the first day has no cases)
covid = [zeros(size(covid,1), 1 ) covid(:,2:end) - covid(:,1:end-1)];
covid(covid<=0) = 0;

% Train & Test Split
% change the name
d_pred_start = strrep(d_pred_start,'-','_');
%d_pred_end = strrep(d_pred_end,'-','_');
%
n_tr_end = find(strcmpi(NYT.Properties.VariableNames, {d_pred_start} )) - 6;
covid_tr = covid(:, 1:n_tr_end);
covid_te = covid(:, n_tr_end+1:n_tr_end+delta);
%

% Get number of counties and number of days
[n_cty, n_day_tr]=size(covid_tr);
[~    , n_day_te]=size(covid_te);


end

