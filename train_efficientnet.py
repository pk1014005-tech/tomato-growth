import os
import sys
import io
import logging
import random
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from contextlib import redirect_stdout

# EfficientNet-B0
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# =========================
# 경로(고정 구조)
# =========================
BASE_DIR = "/root/nymc_Project/data/Tomato1241(6,2,2)"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")
TEST_DIR  = os.path.join(BASE_DIR, "test")

CSV_TRAIN = os.path.join(TRAIN_DIR, "_train.csv")
CSV_VALID = os.path.join(VAL_DIR,   "_val.csv")
CSV_TEST  = os.path.join(TEST_DIR,  "_test.csv")

PROJECT_ROOT = "/root/nymc_Project"
OUT_DIR_MODELS = os.path.join(PROJECT_ROOT, "models")
OUT_DIR_LOGS   = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(OUT_DIR_MODELS, exist_ok=True)
os.makedirs(OUT_DIR_LOGS,   exist_ok=True)

BEST_PATH   = os.path.join(OUT_DIR_MODELS, "best_model.pth")
LAST_PATH   = os.path.join(OUT_DIR_MODELS, "last_model.pth")
CURVE_PATH  = os.path.join(OUT_DIR_LOGS,   "training_curves.png")
CM_PATH     = os.path.join(OUT_DIR_LOGS,   "confusion_matrix.png")
CLFCSV_PATH = os.path.join(OUT_DIR_LOGS,   "classification_report.csv")
HIST_PATH   = os.path.join(OUT_DIR_LOGS,   "history.csv")
LOG_PATH    = os.path.join(OUT_DIR_LOGS,   "train.log")
MODEL_SUMMARY_PATH = os.path.join(OUT_DIR_LOGS, "model_summary.txt")
MODEL_PARAMS_CSV   = os.path.join(OUT_DIR_LOGS, "model_parameters.csv")
F1_SUMMARY_CSV     = os.path.join(OUT_DIR_LOGS, "f1_summary.csv")
BEST_SUMMARY_TXT   = os.path.join(OUT_DIR_LOGS, "best_score.txt")

# =========================
# 로깅 헬퍼
# =========================
class _StreamToLogger(io.TextIOBase):
    def __init__(self, level_func):
        self.level_func = level_func
    def write(self, buf):
        for line in buf.splitlines():
            if line.strip():
                self.level_func(line)
        return len(buf)
    def flush(self): pass

def setup_logging():
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger("").addHandler(console)
    # print()를 파일로 복제
    sys.stdout = _StreamToLogger(logging.info)

# =========================
# 재현성
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================
# 하이퍼파라미터
# =========================
image_size   = 224
batch_size   = 64
epochs       = 160
patience     = 20
lr           = 1e-3
weight_decay = 5e-4
augment_count = 1

# EfficientNet 옵션
USE_PRETRAINED = False            # True면 ImageNet 가중치 + 정규화 자동 적용
EFF_DROPOUT    = 0.5

# H100 최적화 스위치
USE_BF16          = True
USE_CHANNELS_LAST = True
USE_COMPILE       = True

# =========================
# 유틸
# =========================
def _validate_csv(csv_path: str, img_dir: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV가 없습니다: {csv_path}")
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError(f"CSV에 'filename' 열이 필요합니다: {csv_path}")
    exists = df["filename"].astype(str).apply(lambda f: os.path.exists(os.path.join(img_dir, f)))
    if (~exists).any():
        miss = df.loc[~exists, "filename"].astype(str).tolist()
        logging.warning(f"[{os.path.basename(csv_path)}] CSV엔 있으나 파일 없는 항목 {len(miss)}개 (예: {miss[:5]})")
    label_cols = [c for c in df.columns if c != "filename"]
    try:
        df[label_cols] = df[label_cols].apply(pd.to_numeric, errors="raise")
    except Exception:
        df[label_cols] = df[label_cols].apply(pd.to_numeric, errors="coerce")
        if df[label_cols].isna().any().any():
            bad = df[df[label_cols].isna().any(axis=1)].head(5)
            raise ValueError(f"[{os.path.basename(csv_path)}] 라벨에 숫자 외 값 포함 (예시 상위5개):\n{bad}")
    return df

# =========================
# Dataset
# =========================
class TomatoDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, augment_count=1):
        self.data = _validate_csv(csv_path, img_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.augment_count = max(1, int(augment_count))
        self.class_names = list(self.data.columns[1:])
        self.img_names = self.data["filename"].astype(str).tolist()
        self.labels = self.data.drop("filename", axis=1).values.astype("float32")

    def __len__(self):
        return len(self.img_names) * self.augment_count

    def __getitem__(self, idx):
        true_idx = idx % len(self.img_names)
        img_path = os.path.join(self.img_dir, self.img_names[true_idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image) if self.transform else transforms.ToTensor()(image)
        label = torch.tensor(self.labels[true_idx], dtype=torch.float32)
        return image, label

# =========================
# 전처리(사전학습 시 ImageNet 정규화)
# =========================
def build_transforms():
    if USE_PRETRAINED:
        _w = EfficientNet_B0_Weights.IMAGENET1K_V1
        mean, std = _w.transforms().mean, _w.transforms().std
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    train_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_tfm, eval_tfm

# =========================
# EfficientNet-B0 커스텀
# =========================
class CustomEfficientNetB0(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False, dropout: float = 0.5):
        super().__init__()
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            backbone = efficientnet_b0(weights=weights)
        else:
            backbone = efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# =========================
# 모델 요약/파라미터 저장
# =========================
def save_model_summaries(m: nn.Module, num_classes: int, sample_size=(3, image_size, image_size)):
    with open(MODEL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("# Model Architecture (repr)\n\n")
        f.write(repr(m)); f.write("\n\n")
        try:
            from torchsummary import summary
            tmp = CustomEfficientNetB0(num_classes=num_classes, pretrained=False, dropout=EFF_DROPOUT).to("cpu")
            buf = io.StringIO()
            with redirect_stdout(buf):
                summary(tmp, input_size=sample_size)
            f.write("# torchsummary\n\n")
            f.write(buf.getvalue()); f.write("\n")
            del tmp
        except Exception as e:
            f.write(f"[info] torchsummary unavailable: {e}\n")
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        f.write(f"# Parameters\nTotal: {total:,}\nTrainable: {trainable:,}\n")

    rows = []
    for name, p in m.named_parameters():
        rows.append({
            "name": name,
            "shape": list(p.shape),
            "numel": int(p.numel()),
            "requires_grad": bool(p.requires_grad)
        })
    pd.DataFrame(rows).to_csv(MODEL_PARAMS_CSV, index=False)
    print(f"[model] summary saved → {MODEL_SUMMARY_PATH}")
    print(f"[model] parameters saved → {MODEL_PARAMS_CSV}")

# =========================
# 학습/평가 루틴
# =========================
def train_validate(model, train_loader, valid_loader, device, epochs, patience, optimizer, scheduler, criterion):
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False)
        for step, (images, labels) in enumerate(pbar, start=1):
            images = images.to(device, non_blocking=True)
            if USE_CHANNELS_LAST:
                images = images.to(memory_format=torch.channels_last)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(outputs).to(torch.float32)
            preds = (probs > 0.5).to(torch.int32).cpu().numpy()
            true  = labels.to(torch.int32).cpu().numpy()
            train_correct += (preds == true).sum()
            train_total   += true.size
            pbar.set_postfix(loss=f"{(total_loss/step):.4f}")

        avg_train_loss = total_loss / max(1, len(train_loader))
        train_acc = train_correct / max(1, train_total)

        # ---- Validation ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device, non_blocking=True)
                if USE_CHANNELS_LAST:
                    images = images.to(memory_format=torch.channels_last)
                labels = labels.to(device, dtype=torch.float32, non_blocking=True)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                probs = torch.sigmoid(outputs).to(torch.float32)
                preds = (probs > 0.5).to(torch.int32).cpu().numpy()
                true  = labels.to(torch.int32).cpu().numpy()
                val_correct += (preds == true).sum()
                val_total   += true.size

        avg_val_loss = val_loss / max(1, len(valid_loader))
        val_acc = val_correct / max(1, val_total)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(avg_val_loss)

        # 체크포인트 & EarlyStopping
        torch.save(model.state_dict(), LAST_PATH)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), BEST_PATH)
            print(f"[checkpoint] best updated: {best_val_acc:.4f}")
            with open(BEST_SUMMARY_TXT, "w", encoding="utf-8") as f:
                f.write(f"best_val_acc: {best_val_acc:.6f}\n")
                f.write(f"epoch: {epoch+1}\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⛔ Early stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
                break

    return history, best_val_acc

def test_and_report(model, test_loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            if USE_CHANNELS_LAST:
                images = images.to(memory_format=torch.channels_last)
            outputs = model(images)
            preds = (torch.sigmoid(outputs).to(torch.float32).cpu().numpy() > 0.5).astype(int)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Top-1 참고용 혼동행렬
    true_cls = np.argmax(all_labels, axis=1)
    pred_cls = np.argmax(all_preds, axis=1)
    cm = confusion_matrix(true_cls, pred_cls)

    plt.figure(figsize=(8, 6))
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
    except Exception:
        plt.imshow(cm, cmap='Blues'); plt.colorbar()
        plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45, ha='right')
        plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
        for (i,j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha='center', va='center')
    plt.title("Test Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(CM_PATH); plt.close()

    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    print("\n===== Test Set Classification Report =====")
    print(report_df.round(4))
    report_df.to_csv(CLFCSV_PATH)

    rows = []
    for key in ["micro avg", "macro avg", "weighted avg"]:
        if key in report_dict:
            rows.append({
                "metric": key,
                "precision": report_dict[key].get("precision", np.nan),
                "recall":    report_dict[key].get("recall",    np.nan),
                "f1":        report_dict[key].get("f1-score",  np.nan),
                "support":   report_dict[key].get("support",   np.nan),
            })
    acc = report_dict.get("accuracy", np.nan)
    rows.append({"metric": "accuracy", "precision": np.nan, "recall": np.nan, "f1": acc, "support": np.sum(all_labels)})
    pd.DataFrame(rows).to_csv(F1_SUMMARY_CSV, index=False)
    print(f"[result] F1 summary saved → {F1_SUMMARY_CSV}")

# =========================
# 메인
# =========================
def main():
    setup_logging()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Training start ===")
    print(f"BASE_DIR: {BASE_DIR}")
    print("Device:", device); logging.info(f"Device: {device}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # 전처리
    train_tfm, eval_tfm = build_transforms()

    # 데이터로더
    train_ds = TomatoDataset(CSV_TRAIN, TRAIN_DIR, train_tfm, augment_count=augment_count)
    valid_ds = TomatoDataset(CSV_VALID, VAL_DIR,   eval_tfm)
    test_ds  = TomatoDataset(CSV_TEST,  TEST_DIR,  eval_tfm)
    class_names = train_ds.class_names

    pin = torch.cuda.is_available()
    num_workers = max(8, (os.cpu_count() or 16) // 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, persistent_workers=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin, persistent_workers=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin, persistent_workers=False)

    # 모델
    num_classes = len(class_names)
    model = CustomEfficientNetB0(num_classes=num_classes, pretrained=USE_PRETRAINED, dropout=EFF_DROPOUT)
    model = model.to(device)
    if USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)

    if USE_COMPILE:
        try:
            model = torch.compile(model, mode="max-autotune")
            print("[compile] torch.compile 활성화")
        except Exception as e:
            print(f"[compile] 사용 불가: {e}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 모델 요약 저장
    save_model_summaries(model, num_classes=num_classes)

    # 학습 + 검증
    history, best_val_acc = train_validate(
        model, train_loader, valid_loader, device, epochs, patience, optimizer, scheduler, criterion
    )

    # 곡선/히스토리 저장
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label='Train Loss')
    plt.plot(history["val_loss"], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label='Train Acc')
    plt.plot(history["val_acc"], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy Curve'); plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH)
    plt.close()
    pd.DataFrame(history).to_csv(HIST_PATH, index=False)

    # 테스트(베스트 모델 로드)
    model.load_state_dict(torch.load(BEST_PATH, map_location=device))
    model = model.to(device)
    if USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    test_and_report(model, test_loader, device, class_names)

    print("=== Training done ===")
    print(f"Saved: {BEST_PATH}, {LAST_PATH}")
    print(f"Logs : {CURVE_PATH}, {CM_PATH}, {CLFCSV_PATH}, {HIST_PATH}, {LOG_PATH}, {MODEL_SUMMARY_PATH}, {MODEL_PARAMS_CSV}, {F1_SUMMARY_CSV}, {BEST_SUMMARY_TXT}")

if __name__ == "__main__":
    main()
