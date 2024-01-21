from github import Github
import os


def my_upload(token):
    g = Github(token)

    repo_name = "kaggle_output"
    repo = g.get_user().get_repo(repo_name)

    # 定义文件夹和文件类型的映射关系
    folder_mapping = {
        ".txt": "log_files",
        ".pth": "model_files",
        ".jpg": "los_image_files"
    }

    local_folder = "/kaggle/working/"
    for file_name in os.listdir(local_folder):
        if os.path.isfile(os.path.join(local_folder, file_name)):
            file_extension = os.path.splitext(file_name)[1].lower()

            # 确保文件类型在映射关系中
            if file_extension in folder_mapping:
                folder_name = folder_mapping[file_extension]

                # 获取目标文件夹中现有文件的数量
                existing_files = repo.get_contents(folder_name)
                num_files = len(existing_files)

                # 构建新文件的名称
                new_file_name = f"{file_name.split('.')[0]}{num_files}{file_extension}"
                remote_file_path = f"{folder_name}/{new_file_name}"

                # 读取本地文件的内容
                with open(os.path.join(local_folder, file_name), 'rb') as file:
                    file_content = file.read()

                # 在仓库中创建或更新文件
                repo.create_file(remote_file_path, f"Upload {remote_file_path}", file_content, branch="main")

    print("Files uploaded successfully.")
