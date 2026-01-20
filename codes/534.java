public class SupplementingParser extends AbstractMultipleParser {
    /**
     * Serial version UID.
     */
    private static final long serialVersionUID = 313179254565350994L;

    /**
     * The different Metadata Policies we support (not discard)
     */
    public static final List<MetadataPolicy> allowedPolicies =
            Arrays.asList(MetadataPolicy.FIRST_WINS,
                          MetadataPolicy.LAST_WINS,
                          MetadataPolicy.KEEP_ALL);

    @SuppressWarnings("rawtypes")
    public SupplementingParser(MediaTypeRegistry registry,
                               Collection<? extends Parser> parsers, Map<String, Param> params) {
        super(registry, parsers, params);
    }
    public SupplementingParser(MediaTypeRegistry registry, MetadataPolicy policy,
                               Parser... parsers) {
        this(registry, policy, Arrays.asList(parsers));
    }
    public SupplementingParser(MediaTypeRegistry registry, MetadataPolicy policy,
                               Collection<? extends Parser> parsers) {
        super(registry, policy, parsers);
        
        // Ensure it's a supported policy
        if (!allowedPolicies.contains(policy)) {
            throw new IllegalArgumentException("Unsupported policy for SupplementingParser: " + policy);
        }
    }

    @Override
    protected boolean parserCompleted(Parser parser, Metadata metadata,
            ContentHandler handler, ParseContext context, Exception exception) {
        // If there was no exception, just carry on to the next
        if (exception == null) return true;
        
        // Have the next parser tried
        return true;
    }
}