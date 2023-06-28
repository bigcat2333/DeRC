log_name = "DeRC_layer4_QQP.log"
with open(log_name, "r") as f:
    lines = f.readlines()
    log_lines = []
    for line in lines:
        if " - INFO - step:" in line:
            log_lines.append(line)
    
    loss_lower_anti_bias_loss = []
    loss_lower_bias_loss = []
    loss_upper_anti_bias_loss = []
    loss_upper_bias_loss = []

    for line in log_lines:
        loss_lower_anti_bias_loss_idx = line.find("loss_lower_anti_bias_loss:") + len("loss_lower_anti_bias_loss:")
        loss_lower_bias_loss_idx = line.find("loss_lower_bias_loss:") + len("loss_lower_bias_loss:")
        loss_upper_anti_bias_loss_idx = line.find("loss_upper_anti_bias_loss:") + len("loss_upper_anti_bias_loss:")
        loss_upper_bias_loss_idx = line.find("loss_upper_bias_loss:") + len("loss_upper_bias_loss:")

        try:
            now_loss_lower_anti_bias_loss = float(line[loss_lower_anti_bias_loss_idx:loss_lower_anti_bias_loss_idx+7])
            now_loss_lower_bias_loss = float(line[loss_lower_bias_loss_idx:loss_lower_bias_loss_idx+7])
            now_loss_upper_anti_bias_loss = float(line[loss_upper_anti_bias_loss_idx:loss_upper_anti_bias_loss_idx+7])
            now_loss_upper_bias_loss = float(line[loss_upper_bias_loss_idx:loss_upper_bias_loss_idx+7])
        except:
            continue

        loss_lower_anti_bias_loss.append(now_loss_lower_anti_bias_loss)
        loss_lower_bias_loss.append(now_loss_lower_bias_loss)
        loss_upper_anti_bias_loss.append(now_loss_upper_anti_bias_loss)
        loss_upper_bias_loss.append(now_loss_upper_bias_loss)
    
    for item in loss_upper_bias_loss:
        print(item)

