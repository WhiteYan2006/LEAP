from pathlib import Path
import csv,importlib.util,json,subprocess,sys,tempfile,unittest

ROOT=Path(__file__).resolve().parents[1]

def csv_rows(path):
    with Path(path).open(encoding='utf8',newline='') as f:
        return list(csv.DictReader(f))

class PaperDataTests(unittest.TestCase):
    def test_verified_aggregate_inputs(self):
        spec=importlib.util.spec_from_file_location('reproduce',ROOT/'scripts/reproduce_figures.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        r=module.load_verified()
        self.assertEqual([len(r[n]) for n in ['figure5_ood.csv','figure6_shared1024.csv','figure7_cpu.csv','figure8_overlap.csv','baseline_training.csv']],[39,20,42,12,151])

    def test_replot_matches_all_reported_numeric_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'figures'
            subprocess.run([sys.executable,'-B',str(ROOT/'scripts/reproduce_figures.py'),'--output',str(out)],check=True,capture_output=True)
            plotted=csv_rows(out/'PLOT_VALUES.csv')
            fig6=[r for r in plotted if r['figure']=='figure6'];fig7=[r for r in plotted if r['figure']=='figure7']
            self.assertEqual(len(fig6),80);self.assertEqual(len(fig7),126)
            self.assertEqual(sum(r['figure']=='figure5' for r in plotted),27)
            self.assertEqual(sum(r['figure']=='training' for r in plotted),151)
            names={'fcfs':'FCFS','lie_sqrt':'LEAP','tie_expectation':'TIE','ltr_cal':'LTR'}
            for r in csv_rows(ROOT/'paper_data/figure6_shared1024.csv'):
                for key in ['mean_E2E_per_output_token_s','p90_E2E_per_output_token_s','mean_TTFT_s','p90_TTFT_s']:
                    q=next(q for q in fig6 if q['method']==names[r['arm']] and q['panel']==key and float(q['x'])==int(r['rate_rps']))
                    self.assertEqual(float(q['y']),float(r[key]))
            names={'lie':'LEAP','trail':'TRAIL','tie':'TIE','ltr':'LTR','ssjf':'SSJF','egtp_hidden_fp32':'EGTP-P','egtp_logits':'EGTP-L'}
            for r in csv_rows(ROOT/'paper_data/figure7_cpu.csv'):
                for key in ['delivered_predictions_s','p95_delivery_ms','predict_and_consume_cpu_us_per_prediction']:
                    q=next(q for q in fig7 if q['method']==names[r['method']] and q['panel']==key and float(q['x'])==float(r['scheduled_effective_predictions_s']))
                    self.assertEqual(float(q['y']),float(r[key]))

if __name__=='__main__':unittest.main()

