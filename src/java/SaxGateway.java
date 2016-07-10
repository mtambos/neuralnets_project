import com.opencsv.CSVReader;
import net.seninp.gi.GIAlgorithm;
import net.seninp.gi.logic.GrammarRules;
import net.seninp.gi.logic.RuleInterval;
import net.seninp.gi.repair.RePairFactory;
import net.seninp.gi.repair.RePairGrammar;
import net.seninp.gi.sequitur.SAXRule;
import net.seninp.gi.sequitur.SequiturFactory;
import net.seninp.grammarviz.logic.PackedRuleRecord;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;
import net.seninp.jmotif.sax.SAXProcessor;
import net.seninp.jmotif.sax.alphabet.NormalAlphabet;
import net.seninp.jmotif.sax.datastructure.SAXRecords;
import net.seninp.jmotif.sax.parallel.ParallelSAXImplementation;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;


@SuppressWarnings("WeakerAccess")
public class SaxGateway{
    private LogLevel level = LogLevel.INFO;

    @SuppressWarnings("unused")
    public SaxGateway(){
        this.log("Created");
    }

    @SuppressWarnings("unused")
    public SaxGateway(LogLevel level){
        this.log("Created");
        this.level = level;
    }

    /**
     * Performs logging messages distribution.
     *
     * @param message the message to log.
     */
    private void log(String message) {
        this.log(message, LogLevel.DEBUG);
    }

    /**
     * Performs logging messages distribution.
     *
     * @param message the message to log.
     */
    private void log(String message, LogLevel level) {
        if(this.level.ordinal() <= level.ordinal()){
            System.out.println(message);
        }
    }


    private double[][] parseCsv(String csvFileName) throws IOException {
        CSVReader csvReader = new CSVReader(new FileReader(csvFileName));
        List<String[]> content = csvReader.readAll();

        int N = content.size();
        int M = content.get(0).length;
        double[][] ts = new double[M][N];
        for (int i=0; i < N; i++) {
            String[] row = content.get(i);
            for(int j=0; j < M; j++){
                ts[j][i] = Double.parseDouble(row[j]);
            }
        }
        return ts;
    }

    private SAXRecords mixSaxRecords(SAXRecords[] saxFrequencyData, MixingStrategy mixingStrategy,
                                     int paaSize) {
        SAXRecords mixedFrequencyData = new SAXRecords();
        HashMap<Integer, ArrayList<char[]>> mixedRecords = new HashMap<>();

        for (SAXRecords records : saxFrequencyData) {
            for (Integer index : records.getIndexes()) {
                if(!mixedRecords.containsKey(index)) {
                    mixedRecords.put(index, new ArrayList<>());
                }
                mixedRecords.get(index).add(records.getByIndex(index).getPayload());
            }
        }
        for (Integer index: mixedRecords.keySet()) {
            char[] mixedCode = new char[saxFrequencyData.length*paaSize];

            ArrayList<char[]> code = mixedRecords.get(index);
            for (int i=0; i < code.size(); i++) {
                char[] partialCode = code.get(i);
                for (int j=0; j < partialCode.length; j++) {
                    char symbol = partialCode[j];
                    if (mixingStrategy.equals(MixingStrategy.STACKED)) {
                        mixedCode[i*partialCode.length + j] = symbol;
                    } else {
                        mixedCode[i + j*code.size()] = symbol;
                    }
                }
            }

            mixedFrequencyData.add(mixedCode, index);
        }

        return mixedFrequencyData;
    }

    @SuppressWarnings("unused")
    public synchronized SaxChartData processData(String csvFileName, int mixStrategy, int algorithm,
                                                 boolean useSlidingWindow, int numRedStrategy, int windowSize,
                                                 int paaSize, int alphabetSize, double normalizationThreshold,
                                                 double thresholdLength, double thresholdCom)
            throws Exception {
        double[][] multiTs = parseCsv(csvFileName);
        return this.processData(multiTs, mixStrategy, algorithm, useSlidingWindow, numRedStrategy, windowSize,
                                paaSize, alphabetSize, normalizationThreshold, thresholdLength, thresholdCom);
    }

    /**
     * Process data with Sequitur. Populate and broadcast ChartData object.
     *
     * @param algorithm              the algorithm, 0 Sequitur, 1 RE-PAIR.
     * @param useSlidingWindow       The use sliding window parameter.
     * @param numRedStrategy         The numerosity reduction strategy.
     * @param windowSize             The SAX sliding window size.
     * @param paaSize                The SAX PAA size.
     * @param alphabetSize           The SAX alphabet size.
     * @param normalizationThreshold The normalization threshold.
     */
    public synchronized SaxChartData processData(double[][] multiTs, int mixStrategy, int algorithm,
                                                 boolean useSlidingWindow, int numRedStrategy, int windowSize,
                                                 int paaSize, int alphabetSize, double normalizationThreshold,
                                                 double thresholdLength, double thresholdCom)
            throws Exception
    {

        // check if the data is loaded
        //
        SaxChartData saxData;
        MixingStrategy mixingStrategy = MixingStrategy.fromValue(mixStrategy);
        NumerosityReductionStrategy numerosityReductionStrategy
                = NumerosityReductionStrategy.fromValue(numRedStrategy);
        GIAlgorithm giAlgorithm = GIAlgorithm.fromValue(algorithm);

        // the logging block
        //
        StringBuilder sb = new StringBuilder("setting up GI with params: ");
        if (GIAlgorithm.SEQUITUR.equals(giAlgorithm)) {
            sb.append("algorithm: Sequitur, ");
        } else {
            sb.append("algorithm: RePair, ");
        }
        sb.append("sliding window ").append(useSlidingWindow);
        sb.append(", numerosity reduction ").append(numerosityReductionStrategy.toString());
        sb.append(", SAX window ").append(windowSize);
        sb.append(", PAA ").append(paaSize);
        sb.append(", Alphabet ").append(alphabetSize);
        this.log(sb.toString(), LogLevel.INFO);

        this.log("creating ChartDataStructure", LogLevel.INFO);

        int M = multiTs.length;
        int N = multiTs[0].length;
        SAXRecords[] saxFrequencyData = new SAXRecords[M];
        for(int j=0; j < M; j++) {
            double[] ts = multiTs[j];
            if(useSlidingWindow){ //ParallelSAXImplementation.process discretizes the time series using a sliding window
                ParallelSAXImplementation ps = new ParallelSAXImplementation();
                saxFrequencyData[j] = ps.process(ts, 4, windowSize, paaSize, alphabetSize,
                        numerosityReductionStrategy, normalizationThreshold);
            }else{
                SAXProcessor ps = new SAXProcessor();
                NormalAlphabet na = new NormalAlphabet();
                saxFrequencyData[j] = ps.ts2saxByChunking(ts, paaSize, na.getCuts(alphabetSize), 0.05);
            }
            this.log(String.valueOf(j) + ": " + saxFrequencyData[j].getSAXString(" "));
        }

        SAXRecords mixedSaxRecords = mixSaxRecords(saxFrequencyData, mixingStrategy, paaSize);

        GrammarRules rules;
        if (GIAlgorithm.SEQUITUR.equals(giAlgorithm)) {

            this.log("running sequitur ...");
            SAXRule sequiturGrammar = SequiturFactory.runSequitur(mixedSaxRecords.getSAXString(" "));
            this.log("mixed: " + mixedSaxRecords.getSAXString(" "));

            this.log("collecting grammar rules data ...", LogLevel.INFO);
            rules = sequiturGrammar.toGrammarRulesData();

            this.log("mapping rule intervals on timeseries ...", LogLevel.INFO);
            SequiturFactory.updateRuleIntervals(rules, mixedSaxRecords, useSlidingWindow, new double[N],
                                                windowSize, paaSize);

        } else {


            RePairGrammar rePairGrammar = RePairFactory.buildGrammar(mixedSaxRecords);

            rePairGrammar.expandRules();
            rePairGrammar.buildIntervals(mixedSaxRecords, new double[N], windowSize);

            rules = rePairGrammar.toGrammarRulesData();

        }
        this.log("done ...", LogLevel.INFO);
        saxData = new SaxChartData(multiTs, useSlidingWindow, numerosityReductionStrategy,
                                   windowSize, paaSize, alphabetSize);
        saxData.setGrammarRules(rules);

        this.log("process finished", LogLevel.INFO);

        saxData.performRemoveOverlapping(thresholdLength, thresholdCom);
        return saxData;
    }

    @SuppressWarnings("unused")
    public static HashMap<Integer, int[][]> getPositionsForClasses(SaxChartData saxData) {
        HashMap<Integer, int[][]> positionsForClasses = new HashMap<>();
        for (PackedRuleRecord record : saxData.getArrPackedRuleRecords()) {
            Integer classId = record.getClassIndex();
            ArrayList<RuleInterval> arrPos = saxData.getSubsequencesPositionsByClassNum(classId);
            int[][] positions = new int[arrPos.size()][2];
            for (int i=0; i < arrPos.size(); i++){
                RuleInterval saxPos = arrPos.get(i);
                positions[i][0] = saxPos.getStart();
                positions[i][1] = saxPos.getEnd();
            }
            Arrays.sort(positions, (int[] o1, int[] o2) -> Integer.compare(o1[0], o2[0]));
            positionsForClasses.put(classId, positions);
        }
        return positionsForClasses;
    }
}
