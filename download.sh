# export HF_ENDPOINT=https://hf-mirror.com

# 重试下载，直到成功
RETRY_DELAY=10  # 重试间隔（秒）
MAX_RETRIES=0   # 0 表示无限重试

retry_count=0
while true; do
  echo "开始下载... (尝试次数: $((retry_count + 1)))"
  
  huggingface-cli download billzhao1030/all_scene \
    --repo-type dataset \
    --local-dir /cpfs/shared/simulation/wangliuyi/vlnverse_scene \
    --resume-download \
    --include kujiale_0157.tar

  
  if [ $? -eq 0 ]; then
    echo "下载成功完成！"
    break
  else
    retry_count=$((retry_count + 1))
    if [ $MAX_RETRIES -gt 0 ] && [ $retry_count -ge $MAX_RETRIES ]; then
      echo "达到最大重试次数 ($MAX_RETRIES)，退出"
      exit 1
    fi
    echo "下载失败，等待 ${RETRY_DELAY} 秒后重试..."
    sleep $RETRY_DELAY
  fi
done

# export HF_ENDPOINT=https://hf-mirror.com

# huggingface-cli download InternRobotics/InternVLA-N1 \
#   --repo-type model \
#   --local-dir /cpfs/user/wangliuyi/code/vlnverse_vlnverse/checkpoints/InternVLA-N1 \
#   --resume-download 