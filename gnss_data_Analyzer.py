import matplotlib.pyplot as plt
import pandas as pd

# טעינת קובץ ה-GNSS
df = pd.read_csv("device_gnss_p4_train.csv")

# בדיקת מספר שורות ועמודות (לוודא מעל 1,000 שורות ו-10 עמודות)
rows, cols = df.shape
print(f"Number of rows: {rows}")
print(f"Number of columns: {cols}")

# הצגת קורלציה מהירה בין C/N0 לאלווציה וטווח מדומה
columns_to_check = [
    "Cn0DbHz",
    "SvElevationDegrees",
    "RawPseudorangeMeters",
    "AzimuthDegrees",
]


# Set visual style for professional appearance


# ==========================================
# 1. Histogram of C/N0 Distribution
# ==========================================
plt.figure(figsize=(10, 5))
c_t_n0 = df["Cn0DbHz"]
plt.hist(c_t_n0, bins=40, color="royalblue")

plt.title("Distribution of Satellite Signal Strength (C/N0)", fontsize=14, pad=15)
plt.xlabel("C/N0 [dB-Hz]", fontsize=12)
plt.ylabel("Frequency (Number of Measurements)", fontsize=12)

plt.tight_layout()
plt.show()

# ==========================================
# 2. Scatterplot: C/N0 vs. Elevation Angle
# ==========================================
plt.figure(figsize=(10, 6))
# Using a sample of 5,000 rows to keep the plot clean and readable


plt.title(
    "Satellite Signal Strength (C/N0) vs. Elevation Angle", fontsize=14, pad=15
)
plt.scatter(df["SvElevationDegrees"],df["Cn0DbHz"])
plt.xlabel("Elevation Angle [Degrees]", fontsize=12)
plt.ylabel("C/N0 [dB-Hz]", fontsize=12)

plt.tight_layout()
plt.show()

plt.title(
    "Satellite Signal Strength (C/N0)  vs.PR when Elevation Angle == 0 ", fontsize=14, pad=15
)
filtered_df = df[
    (df["SvElevationDegrees"] >= 0) & (df["SvElevationDegrees"] < 1)
]
# 2. שמירת העמודות למשתנים נפרדים (כ-Series או Lists)
new_pr = filtered_df["RawPseudorangeMeters"]
new_c_t_n = filtered_df["Cn0DbHz"]
plt.scatter(new_pr,new_c_t_n)
plt.xlabel("Elevation Angle [Degrees]", fontsize=12)
plt.ylabel("C/N0 [dB-Hz]", fontsize=12)

plt.tight_layout()
plt.show()

gps_l1 = df[(df["ConstellationType"] == 1) & (df["SignalType"] == "GPS_L1_CA")].copy()

# 2. בחירת הלוויין (Svid) שיש לו הכי הרבה מדידות לאורך זמן בקובץ
best_svid = gps_l1["Svid"].value_counts().index[0]
single_sat = gps_l1[gps_l1["Svid"] == best_svid].sort_values("ArrivalTimeNanosSinceGpsEpoch")

# 3. יצירת ציר זמן בשניות (מתחיל מ-0)
time_seconds = (
    single_sat["ArrivalTimeNanosSinceGpsEpoch"] - single_sat["ArrivalTimeNanosSinceGpsEpoch"].iloc[0]
) / 1000000.0
single_sat["TimeSec"] = time_seconds

# 4. חישוב "קו ה-LOS האידיאלי" בעזרת ממוצע נע (Rolling Mean של 10 שניות)
single_sat["Smooth_LOS"] = (
    single_sat["Cn0DbHz"].rolling(window=10, min_periods=1, center=True).mean()
)

# 5. שרטוט הגרף
plt.figure(figsize=(14, 6))

# הקו החלק - מייצג את עקומת ה-LOS הבסיסית
plt.plot(
    single_sat["TimeSec"],
    single_sat["Smooth_LOS"],
    color="orange",
    linewidth=2.5,
    label="Baseline / Smooth Trend (Approx. Direct LOS)",
    zorder=3,
)

# הנקודות בפועל - חושפות את ההתאבכויות
plt.scatter(
    single_sat["TimeSec"],
    single_sat["Cn0DbHz"],
    color="dodgerblue",
    alpha=0.7,
    s=25,
    label="Actual Measurements (Multipath Interference)",
    zorder=2,
)

plt.title(
    f"Constructive vs. Destructive Interference Over Time (GPS L1, Svid = {best_svid})",
    fontsize=14,
    pad=15,
)
plt.xlabel("Time [Seconds]", fontsize=12)
plt.ylabel("C/N0 [dB-Hz]", fontsize=12)
plt.legend(fontsize=11, loc="lower left")
plt.grid(True, alpha=0.3)

plt.show()