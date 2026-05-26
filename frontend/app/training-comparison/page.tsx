import type { Metadata } from "next";
import TrainingComparison from "../../components/TrainingComparison";

export const metadata: Metadata = {
    title: "DQN vs PPO MLflow Training Comparison",
    description: "Static MLflow metric-history comparison for DQN and PPO aircraft predictive-maintenance training runs.",
};

export default function TrainingComparisonPage() {
    return <TrainingComparison />;
}
