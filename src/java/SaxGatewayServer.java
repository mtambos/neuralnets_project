import net.seninp.gi.GIAlgorithm;
import net.seninp.grammarviz.logic.PackedRuleRecord;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;
import py4j.DefaultGatewayServerListener;
import py4j.GatewayServer;

import java.util.HashMap;
import java.util.Random;


@SuppressWarnings("WeakerAccess")
public class SaxGatewayServer extends DefaultGatewayServerListener {

    @SuppressWarnings("unused")
    public static void test() throws Exception {
        SaxGateway gs = new SaxGateway(LogLevel.DEBUG);
        double[][] ts = new double[4][1000];
        Random rnd = new Random();
        for (int j = 0; j < ts.length; j++) {
            for (int i = 0; i < ts[j].length; i++) {
                ts[j][i] = rnd.nextDouble();
            }
        }
        SaxChartData saxData =  gs.processData(ts, MixingStrategy.SHUFFLED.ordinal(), GIAlgorithm.SEQUITUR.ordinal(),
                true, NumerosityReductionStrategy.NONE.ordinal(), 42, 4, 8, 0.05, 0.1, 0.5);

        System.out.println(saxData.getSAXDisplay());

        StringBuilder sb = new StringBuilder();
        for (PackedRuleRecord rr : saxData.getArrPackedRuleRecords()) {
            sb.append(rr.getClassIndex());
            sb.append(" ");
            sb.append(rr.getMaxLength());
            sb.append(" ");
            sb.append(rr.getMinLength());
            sb.append(" ");
            sb.append(rr.getSubsequenceNumber());
            sb.append("\n");
        }
        System.out.println(sb.toString());

        sb = new StringBuilder();
        HashMap<Integer, int[][]> positionsForClasses = SaxGateway.getPositionsForClasses(saxData);
        for (Integer classId : positionsForClasses.keySet()) {
            for (int[] d : positionsForClasses.get(classId)) {
                sb.append(classId);
                sb.append(" ");
                sb.append(d[0]);
                sb.append(" ");
                sb.append(d[1]);
                sb.append("\n");
            }
            sb.append("\n\n");
        }
        System.out.println(sb.toString());
    }

    public static void main(String[] args) throws Exception {
        GatewayServer gatewayServer;
//        test();
        if (args.length == 0) {
            gatewayServer = new GatewayServer(new SaxGatewayServer());
        } else {
            gatewayServer = new GatewayServer(new SaxGatewayServer(), Integer.parseInt(args[0]));
            System.out.println(args[0]);
        }
        gatewayServer.start();
        System.out.println("Gateway Server Started: ");
    }
}